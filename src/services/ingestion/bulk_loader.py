"""Bulk ingestion orchestrator.

Coordinates the full pipeline: fetch from CourtListener → parse
metadata → extract + clean text → chunk → store in Postgres.
Processes courts in sequence and opinions in configurable batches.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from src.core.exceptions import IngestionError
from src.models.responses import IngestionProgressResponse
from src.services.ingestion.chunker import chunk_opinion
from src.services.ingestion.parser import extract_best_text, parse_opinion_response

if TYPE_CHECKING:
    from datetime import date

    from src.db.repositories import OpinionRepo
    from src.services.ingestion.courtlistener import CourtListenerClient

logger: structlog.stdlib.BoundLogger = structlog.get_logger()


@dataclass
class _IngestionCounters:
    """Mutable counters accumulated during an ingestion run."""

    fetched: int = 0
    stored: int = 0
    skipped: int = 0
    errors: int = 0
    chunks: int = 0
    court_cache: dict[str, dict[str, Any]] = field(default_factory=dict)


class BulkLoader:
    """Orchestrates opinion ingestion from CourtListener into Postgres."""

    def __init__(
        self,
        cl_client: CourtListenerClient,
        opinion_repo: OpinionRepo,
        *,
        batch_size: int = 20,
    ) -> None:
        self._cl = cl_client
        self._repo = opinion_repo
        self._batch_size = batch_size

    async def ingest(
        self,
        court_ids: list[str],
        *,
        date_after: date | None = None,
        date_before: date | None = None,
        max_opinions: int = 100,
    ) -> IngestionProgressResponse:
        """Run the full ingestion pipeline for the given courts.

        Returns an IngestionProgressResponse summarising what was processed.
        """
        t0 = time.monotonic()
        counters = _IngestionCounters()

        per_court = max(max_opinions // len(court_ids), 1) if court_ids else 0

        for court_id in court_ids:
            await self._ingest_court(
                court_id,
                date_after=date_after,
                date_before=date_before,
                max_opinions=per_court,
                counters=counters,
            )

        elapsed = time.monotonic() - t0
        logger.info(
            "ingestion_complete",
            fetched=counters.fetched,
            stored=counters.stored,
            skipped=counters.skipped,
            errors=counters.errors,
            chunks=counters.chunks,
            elapsed_seconds=round(elapsed, 2),
        )

        return IngestionProgressResponse(
            court_ids=court_ids,
            total_fetched=counters.fetched,
            total_stored=counters.stored,
            total_skipped=counters.skipped,
            total_errors=counters.errors,
            total_chunks=counters.chunks,
            elapsed_seconds=round(elapsed, 2),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ingest_court(
        self,
        court_id: str,
        *,
        date_after: date | None,
        date_before: date | None,
        max_opinions: int,
        counters: _IngestionCounters,
    ) -> None:
        """Fetch and store opinions for a single court."""
        logger.info("ingesting_court", court_id=court_id, max_opinions=max_opinions)

        clusters = await self._cl.fetch_opinions(
            court_id,
            date_after=date_after,
            date_before=date_before,
            max_results=max_opinions,
        )

        court_data = await self._get_court_data(court_id, counters)

        batch: list[dict[str, object]] = []

        for cluster_data in clusters:
            opinion_ids = cluster_data.get("sub_opinions", [])
            if not opinion_ids:
                opinion_ids = [cluster_data.get("id")]

            for oid_ref in opinion_ids:
                opinion_id = self._extract_id(oid_ref)
                if opinion_id is None:
                    continue

                row = await self._process_opinion(opinion_id, cluster_data, court_data, counters)
                if row is not None:
                    batch.append(row)

                if len(batch) >= self._batch_size:
                    await self._flush_batch(batch, counters)

        if batch:
            await self._flush_batch(batch, counters)

    async def _get_court_data(self, court_id: str, counters: _IngestionCounters) -> dict[str, Any]:
        """Fetch court metadata, caching across calls."""
        if court_id in counters.court_cache:
            return counters.court_cache[court_id]
        try:
            data = await self._cl.fetch_court_detail(court_id)
        except IngestionError:
            data = {"id": court_id, "short_name": court_id, "jurisdiction": ""}
        counters.court_cache[court_id] = data
        return data

    async def _process_opinion(
        self,
        opinion_id: int,
        cluster_data: dict[str, Any],
        court_data: dict[str, Any],
        counters: _IngestionCounters,
    ) -> dict[str, object] | None:
        """Fetch, parse, clean, and chunk a single opinion. Returns a row dict or None."""
        try:
            opinion_data = await self._cl.fetch_opinion_detail(opinion_id)
            counters.fetched += 1

            metadata = parse_opinion_response(opinion_data, cluster_data, court_data)
            raw_text = extract_best_text(opinion_data)

            if not raw_text:
                logger.warning("skipping_empty_text", opinion_id=opinion_id)
                counters.errors += 1
                return None

            chunks = chunk_opinion(raw_text, opinion_id)
            counters.chunks += len(chunks)

            return {
                "courtlistener_id": metadata.opinion_id,
                "court_id": metadata.court_id,
                "court_level": metadata.court_level.value,
                "case_name": metadata.case_name,
                "date_filed": metadata.date_filed,
                "precedential_status": metadata.precedential_status.value,
                "raw_text": raw_text,
                "citation_count": metadata.citation_count,
                "judges": metadata.judges,
                "jurisdiction": metadata.jurisdiction,
                "source_url": metadata.source_url,
            }

        except IngestionError as exc:
            logger.warning("opinion_processing_failed", opinion_id=opinion_id, error=str(exc))
            counters.errors += 1
            return None

    async def _flush_batch(
        self,
        batch: list[dict[str, object]],
        counters: _IngestionCounters,
    ) -> None:
        """Bulk-insert a batch of opinion rows."""
        inserted = await self._repo.bulk_create(batch)
        skipped = len(batch) - inserted
        counters.stored += inserted
        counters.skipped += skipped
        logger.debug(
            "batch_flushed",
            batch_size=len(batch),
            inserted=inserted,
            skipped=skipped,
        )
        batch.clear()

    @staticmethod
    def _extract_id(ref: Any) -> int | None:
        """Extract an integer ID from a CL API reference (int or URL string)."""
        if isinstance(ref, int):
            return ref
        if isinstance(ref, str):
            parts = ref.rstrip("/").split("/")
            try:
                return int(parts[-1])
            except (ValueError, IndexError):
                return None
        return None
