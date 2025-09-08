import asyncio
import httpx
import numpy as np
import tiktoken
import time
import random
import re
import tracemalloc

from django.core.management.base import BaseCommand
from asgiref.sync import sync_to_async
from django.db.models.functions import Length
from documents.models import Document, DocumentChunk

# -----------------------------
# Config
# -----------------------------
EMBEDDING_API_URL = "http://localhost:8080/v1/embeddings"
EXPECTED_DIM = 1024  # Dimension for Qwen3-Embedding-0.6B
MAX_CHUNK_TOKENS = 1536  # Max tokens per chunk to send to the model
OVERLAP_TOKENS = 50  # How many tokens to overlap between chunks
MAX_DOC_TOKENS = 1572153  # Stop processing documents longer than this

CONCURRENT_REQUESTS = 2  # Number of parallel requests to the embedding server
BATCH_SIZE = 1  # Number of documents to fetch from DB at a time
HTTP_TIMEOUT_S = 120.0  # Timeout for API requests
RETRIES = 3  # Number of times to retry a failed API request

# -----------------------------
# Tokenizer
# -----------------------------
encoding = tiktoken.get_encoding("cl100k_base")

# -----------------------------
# Profiling Helpers
# -----------------------------
try:
    import psutil
except ImportError:
    psutil = None


def _bytes_to_mb(n: int) -> float:
    return round(n / (1024 * 1024), 2)


def mem_snapshot():
    """Returns (current, peak, rss) memory usage in MB."""
    current, peak = tracemalloc.get_traced_memory()
    rss = psutil.Process().memory_info().rss if psutil else None
    return _bytes_to_mb(current), _bytes_to_mb(peak), (_bytes_to_mb(rss) if rss else None)


# -----------------------------
# Streaming Chunker
# -----------------------------
def iter_token_chunks_stream(
    text: str,
    max_tokens: int = MAX_CHUNK_TOKENS,
    overlap: int = OVERLAP_TOKENS,
    max_doc_tokens: int | None = MAX_DOC_TOKENS,
):
    """Splits text into overlapping chunks based on token count."""
    # Split by paragraphs to respect document structure
    pieces = re.split(r"(\n\s*\n)", text)
    buf_tokens: list[int] = []
    produced_tokens = 0

    for piece in pieces:
        piece_tokens = encoding.encode(piece)
        if max_doc_tokens is not None and produced_tokens >= max_doc_tokens:
            break

        while piece_tokens:
            space_left = max_tokens - len(buf_tokens)
            take = min(space_left, len(piece_tokens))
            buf_tokens.extend(piece_tokens[:take])
            piece_tokens = piece_tokens[take:]

            if len(buf_tokens) >= max_tokens:
                chunk_text = encoding.decode(buf_tokens)
                keep = max(0, min(overlap, len(buf_tokens)))
                buf_tokens = buf_tokens[-keep:]  # Overlap buffer
                produced_tokens += len(encoding.encode(chunk_text))
                if chunk_text.strip():
                    yield chunk_text

    if buf_tokens:
        final_chunk = encoding.decode(buf_tokens)
        if final_chunk.strip():
            yield final_chunk


# -----------------------------
# Synchronous DB helpers (to be wrapped by sync_to_async)
# -----------------------------
def save_chunk_embedding(doc_id, chunk_index, text, embedding):
    """Insert or update a DocumentChunk row with its embedding."""
    DocumentChunk.objects.update_or_create(
        document_id=doc_id,
        chunk_index=chunk_index,
        defaults={"content": text, "embedding": embedding},
    )


def update_doc_embedding(doc_id, embedding):
    """Store the average embedding in the parent Document."""
    Document.objects.filter(id=doc_id).update(embedding=embedding)


# -----------------------------
# Management Command
# -----------------------------
class Command(BaseCommand):
    help = "Generate embeddings via API and store per-chunk, with a doc-level average."

    async def _post_with_retries(self, client: httpx.AsyncClient, payload: dict) -> dict:
        """Makes a POST request to the embedding API with exponential backoff."""
        last_exc = None
        for attempt in range(RETRIES):
            try:
                resp = await client.post(EMBEDDING_API_URL, json=payload, timeout=HTTP_TIMEOUT_S)
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as e:
                self.stderr.write(
                    self.style.WARNING(f"⚠️ HTTP Error {e.response.status_code} on attempt {attempt + 1}/{RETRIES}")
                )
                last_exc = e
            except httpx.RequestError as e:
                self.stderr.write(self.style.WARNING(f"⚠️ Request Error on attempt {attempt + 1}/{RETRIES}: {e}"))
                last_exc = e

            if attempt < RETRIES - 1:
                wait = (2**attempt) + random.uniform(0, 1)
                await asyncio.sleep(wait)

        raise last_exc  # Raise the last exception if all retries fail

    async def embed_doc(self, client: httpx.AsyncClient, doc, semaphore: asyncio.Semaphore):
        """Processes a single document: chunks, embeds, and saves results."""
        async with semaphore:
            t0 = time.perf_counter()
            running_sum = np.zeros(EXPECTED_DIM, dtype=np.float32)
            chunk_count = 0

            try:
                for idx, chunk_text in enumerate(iter_token_chunks_stream(doc.content)):
                    payload = {"input": chunk_text}
                    data = await self._post_with_retries(client, payload)
                    emb = data["data"][0]["embedding"]

                    if len(emb) != EXPECTED_DIM:
                        raise ValueError(f"Expected {EXPECTED_DIM} dims, got {len(emb)}")

                    await sync_to_async(save_chunk_embedding)(doc.id, idx, chunk_text, emb)
                    running_sum += np.asarray(emb, dtype=np.float32)
                    chunk_count += 1

                if chunk_count > 0:
                    final_embedding = (running_sum / float(chunk_count)).tolist()
                    await sync_to_async(update_doc_embedding)(doc.id, final_embedding)
                    self.stdout.write(
                        self.style.SUCCESS(
                            f"✅ Doc {doc.id}: Processed {chunk_count} chunks in {time.perf_counter() - t0:.2f}s."
                        )
                    )
                else:
                    self.stdout.write(self.style.WARNING(f"⚠️ Doc {doc.id}: No content to chunk."))

            except Exception as e:
                self.stderr.write(
                    self.style.ERROR(f"❌ Doc {doc.id} failed after {time.perf_counter() - t0:.2f}s. Error: {e}")
                )

    async def async_handle(self, *args, **options):
        """The main async logic for the command."""
        self.stdout.write("Finding documents that need embeddings...")

        # Create a fixed "to-do list" before we start processing
        qs = (
            Document.objects.filter(chunks__embedding__isnull=True)
            .distinct()
            .annotate(
                content_len=Length("content")  # 1. Calculate content length
            )
            .order_by("content_len")
        )
        documents_to_process = await sync_to_async(list)(qs)
        total = len(documents_to_process)

        if total == 0:
            self.stdout.write(self.style.SUCCESS("All documents already have embeddings."))
            return

        self.stdout.write(f"Found {total} documents to process. Concurrency={CONCURRENT_REQUESTS}.")

        semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
        limits = httpx.Limits(max_connections=CONCURRENT_REQUESTS, max_keepalive_connections=CONCURRENT_REQUESTS)

        async with httpx.AsyncClient(limits=limits) as client:
            tasks = [self.embed_doc(client, doc, semaphore) for doc in documents_to_process]
            await asyncio.gather(*tasks)

        self.stdout.write(self.style.SUCCESS("🎉 Finished embedding all documents."))

    def handle(self, *args, **options):
        """The synchronous entry point for the Django management command."""
        if psutil and not tracemalloc.is_tracing():
            tracemalloc.start()

        try:
            asyncio.run(self.async_handle(*args, **options))
        finally:
            if psutil and tracemalloc.is_tracing():
                cur_mb, peak_mb, rss_mb = mem_snapshot()
                self.stdout.write(
                    self.style.WARNING(
                        f"Profiler summary → Peak memory allocation: {peak_mb}MB | Final RSS: {rss_mb}MB"
                    )
                )
