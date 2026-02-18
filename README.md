# Structured Extraction Pipeline

A production-grade legal document intelligence engine that ingests court opinions, extracts structured intelligence using LLMs, builds a citation graph to detect circuit splits, and exposes it all via a FastAPI service.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         API Layer (FastAPI)                          │
│                                                                      │
│  POST /extract      — submit document(s) for extraction              │
│  GET  /extract/{id} — get extraction status + results                │
│  POST /search       — semantic search over extracted holdings        │
│  GET  /graph/conflicts — get detected circuit splits                 │
│  GET  /graph/authority/{citation} — get citation subgraph            │
│  GET  /health       — dependency health check                        │
│  GET  /metrics      — prometheus metrics                             │
│                                                                      │
│  Middleware: request tracing, structured logging, error handling      │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────────────┐
        │              │                      │
        ▼              ▼                      ▼
┌──────────────┐ ┌───────────────┐  ┌─────────────────┐
│  Ingestion   │ │  Extraction   │  │  Graph Engine    │
│  Service     │ │  Service      │  │                  │
│              │ │               │  │  - Citation      │
│  - Bulk CSV  │ │  - LLM calls  │  │    resolution    │
│  - API poll  │ │  - Schema     │  │  - Conflict      │
│  - Parse     │ │    validation │  │    detection     │
│  - Clean     │ │  - Retry +    │  │  - Authority     │
│  - Chunk     │ │    fallback   │  │    clustering    │
│              │ │  - Confidence │  │  - Subgraph      │
│              │ │    scoring    │  │    queries       │
└──────┬───────┘ └───────┬───────┘  └────────┬────────┘
       │                 │                    │
       ▼                 ▼                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        Data Layer                                    │
│                                                                      │
│  PostgreSQL                                                          │
│  ├── opinions        (raw + metadata)                                │
│  ├── extractions     (structured output per opinion)                 │
│  ├── citations       (edges: citing_opinion → cited_opinion)         │
│  ├── conflicts       (detected circuit splits)                       │
│  └── extraction_jobs (job tracking + status)                         │
│                                                                      │
│  Qdrant                                                              │
│  └── holdings_vectors (embedded extracted holdings for search)       │
│                                                                      │
│  Redis                                                               │
│  ├── extraction queue (job queue for async processing)               │
│  ├── rate limiting    (API + LLM provider limits)                    │
│  └── cache            (repeated citation lookups)                    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Features

- **Structured Extraction** — LLM-powered extraction of holdings, legal standards, dispositions, cited authorities, and dissent/concurrence signals with confidence scoring and Pydantic validation
- **Citation Graph** — Resolves citation strings to database opinion IDs, builds subgraphs via recursive CTEs, ranks authorities by citation count + court level + recency
- **Circuit Split Detection** — Automatically identifies pairs of appellate courts disagreeing on the same legal issue by analyzing overlapping authorities, opposing dispositions, and conflicting citation types
- **MMR Search** — Maximal Marginal Relevance reranking over Qdrant vector search produces diverse result sets that avoid near-duplicate holdings, with hand-rolled cosine similarity matrix computation
- **Async Throughout** — asyncpg, async httpx, async FastAPI, async SQLAlchemy 2.0 — no sync code blocking the event loop
- **Observable** — Structured JSON logging (structlog), Prometheus metrics, health checks with per-dependency latency probes
- **Production-Ready** — Multi-stage Dockerfile, GitHub Actions CI, Terraform for GCP Cloud Run, graceful shutdown, retry logic, rate limiting

---

## Tech Stack

| Component | Choice | Why |
|---|---|---|
| Language | Python 3.12 | Type-safe, modern async ecosystem |
| Package manager | uv | Fast, deterministic dependency resolution |
| Linter/formatter | ruff | Single tool for linting + formatting |
| Type checking | mypy (strict mode) | Full type coverage, no `Any` leaks |
| Web framework | FastAPI | Async, auto-documented, Pydantic-native |
| Validation | Pydantic v2 | Request/response/domain models with zero trust for LLM output |
| Database | PostgreSQL + SQLAlchemy 2.0 async | Relational data with graph queries via recursive CTEs |
| Vector store | Qdrant | Holdings search with no vendor lock-in |
| Cache/Queue | Redis | Job queue for async extraction + caching |
| Embeddings | OpenAI text-embedding-3-large (1536-dim) | High-quality general-purpose embeddings |
| LLM | OpenAI GPT-4o / Anthropic Claude | Structured extraction with provider failover |
| HTTP client | httpx (async) | Modern, typed HTTP client |
| Retry logic | tenacity | Exponential backoff with jitter |
| Logging | structlog | Structured JSON logging |
| Metrics | prometheus-client | Observability |
| CI | GitHub Actions | Lint → Type check → Test |
| Infrastructure | Docker + Terraform (GCP Cloud Run) | Containerized deployment with auto-scaling 0–10 |

---

## Quickstart

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker and Docker Compose

### 1. Clone and configure

```bash
git clone https://github.com/your-org/structured-extraction-pipeline.git
cd structured-extraction-pipeline
cp .env.example .env
# Edit .env with your API keys (OPENAI_API_KEY, COURTLISTENER_API_KEY, etc.)
```

### 2. Start infrastructure

```bash
docker compose -f infrastructure/docker-compose.yml up -d
```

This starts PostgreSQL, Redis, and Qdrant with health checks.

### 3. Install dependencies and run migrations

```bash
uv sync
uv run alembic upgrade head
```

### 4. Start the API server

```bash
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Ingest opinions from CourtListener

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "court_ids": ["scotus"],
    "date_after": "2023-01-01",
    "max_opinions": 20
  }'
```

### 6. Extract structured intelligence

```bash
curl -X POST http://localhost:8000/api/v1/extract \
  -H "Content-Type: application/json" \
  -d '{
    "opinion_ids": [1, 2, 3],
    "priority": "high"
  }'
```

### 7. Search extracted holdings

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "qualified immunity excessive force",
    "k": 5,
    "strategy": "mmr",
    "lambda_mult": 0.7
  }'
```

---

## API Reference

All endpoints are prefixed with `/api/v1`.

### Health & Metrics

#### `GET /health`

Returns infrastructure dependency status.

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "uptime_seconds": 3421.17,
  "dependencies": [
    { "name": "postgresql", "status": "healthy", "latency_ms": 1.23 },
    { "name": "redis", "status": "healthy", "latency_ms": 0.45 },
    { "name": "qdrant", "status": "healthy", "latency_ms": 2.10 }
  ]
}
```

#### `GET /metrics`

Prometheus-format metrics endpoint.

---

### Ingestion

#### `POST /ingest`

Ingest court opinions from CourtListener.

**Request:**
```json
{
  "court_ids": ["scotus", "ca9"],
  "date_after": "2023-01-01",
  "date_before": "2024-01-01",
  "max_opinions": 100
}
```

**Response:**
```json
{
  "court_ids": ["scotus", "ca9"],
  "total_fetched": 87,
  "total_stored": 85,
  "total_skipped": 2,
  "total_errors": 0,
  "total_chunks": 340,
  "elapsed_seconds": 12.45
}
```

---

### Extraction

#### `POST /extract`

Submit opinions for structured extraction. Returns 202 with a job tracking ID.

**Request:**
```json
{
  "opinion_ids": [101, 102, 103],
  "priority": "high",
  "extraction_model": "gpt-4o"
}
```

**Response (202):**
```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "pending",
  "total_opinions": 3,
  "estimated_completion_seconds": 45.0
}
```

#### `GET /extract/{job_id}`

Poll job status and per-opinion extraction results.

**Response:**
```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "completed",
  "total_opinions": 3,
  "processed": 3,
  "failed": 0,
  "results": [
    {
      "opinion_id": 101,
      "case_name": "District of Columbia v. Heller",
      "status": "completed",
      "extraction": {
        "holding": "The Second Amendment protects an individual's right to possess a firearm...",
        "holding_confidence": 0.95,
        "legal_standard": "strict scrutiny",
        "disposition": "affirmed",
        "disposition_confidence": 0.99,
        "key_authorities": [
          {
            "citation_string": "554 U.S. 570",
            "case_name": "United States v. Miller",
            "citation_context": "Cited to establish historical scope of the Second Amendment",
            "citation_type": "follows"
          }
        ],
        "legal_topics": ["second amendment", "individual rights", "gun control"],
        "dissent_present": true,
        "dissent_summary": "Justice Stevens argues the amendment protects only militia-related interests",
        "concurrence_present": false,
        "concurrence_summary": null
      }
    }
  ]
}
```

---

### Search

#### `POST /search`

Semantic search over extracted holdings with optional MMR reranking.

**Request:**
```json
{
  "query": "qualified immunity excessive force by police officers",
  "k": 5,
  "strategy": "mmr",
  "lambda_mult": 0.7,
  "filters": {
    "court_level": "appellate",
    "date_after": "2020-01-01",
    "court_ids": ["ca9", "ca5"]
  }
}
```

**Response:**
```json
{
  "results": [
    {
      "opinion_id": 42,
      "case_name": "Rivas-Villegas v. Cortesluna",
      "court_id": "scotus",
      "date_filed": "2021-10-18",
      "holding": "Officers are entitled to qualified immunity where the constitutional violation was not clearly established...",
      "relevance_score": 0.92,
      "legal_topics": ["qualified immunity", "excessive force", "fourth amendment"]
    }
  ],
  "metrics": {
    "unique_courts": 3,
    "date_range_years": 3.5,
    "avg_relevance_score": 0.85,
    "avg_pairwise_diversity": 0.62
  }
}
```

---

### Graph

#### `GET /graph/conflicts?min_confidence=0.5`

Detect circuit splits across opinions in the database.

**Response:**
```json
{
  "conflicts": [
    {
      "conflict_id": "101-202",
      "topic": "qualified immunity - excessive force",
      "court_a": "ca5",
      "court_b": "ca9",
      "opinion_a": {
        "opinion_id": 101,
        "case_name": "Smith v. City of Dallas",
        "holding": "...",
        "date_filed": "2023-05-12",
        "court": "ca5"
      },
      "opinion_b": {
        "opinion_id": 202,
        "case_name": "Jones v. City of Portland",
        "holding": "...",
        "date_filed": "2023-08-21",
        "court": "ca9"
      },
      "description": "The Fifth Circuit applies qualified immunity broadly while the Ninth Circuit uses a narrower standard...",
      "confidence": 0.78,
      "status": "active",
      "detected_at": "2024-01-15T10:30:00Z"
    }
  ],
  "total": 1
}
```

#### `GET /graph/authority/{citation}?depth=2`

Build a citation authority subgraph for a given citation string or legal topic.

**Response:**
```json
{
  "anchor": {
    "opinion_id": 101,
    "case_name": "District of Columbia v. Heller",
    "citation_string": "554 U.S. 570",
    "court": "scotus",
    "date_filed": "2008-06-26",
    "citation_count": 1247
  },
  "nodes": [
    {
      "opinion_id": 102,
      "case_name": "McDonald v. City of Chicago",
      "citation_string": "561 U.S. 742",
      "court": "scotus",
      "date_filed": "2010-06-28",
      "citation_count": 853
    }
  ],
  "edges": [
    {
      "source_id": 102,
      "target_id": 101,
      "citation_type": "follows",
      "context": "Extending Heller's individual right to bear arms to state and local governments"
    }
  ]
}
```

---

## Project Structure

```
structured-extraction-pipeline/
├── src/
│   ├── api/
│   │   ├── app.py                       # FastAPI app factory + lifespan
│   │   ├── dependencies.py              # Dependency injection providers
│   │   ├── middleware.py                 # Request tracing, error handling
│   │   └── routes/
│   │       ├── extraction.py            # POST /extract, GET /extract/{id}
│   │       ├── graph.py                 # GET /graph/conflicts, /graph/authority
│   │       ├── health.py               # GET /health, GET /metrics
│   │       ├── ingestion.py            # POST /ingest
│   │       └── search.py              # POST /search
│   ├── core/
│   │   ├── config.py                    # Pydantic BaseSettings
│   │   ├── exceptions.py               # Custom exception hierarchy
│   │   └── logging.py                  # structlog configuration
│   ├── models/
│   │   ├── database.py                  # SQLAlchemy ORM models
│   │   ├── domain.py                    # Core domain models (Pydantic v2)
│   │   ├── requests.py                  # API request schemas
│   │   └── responses.py                # API response schemas
│   ├── services/
│   │   ├── extraction/
│   │   │   ├── extractor.py             # Extraction orchestrator
│   │   │   ├── llm_client.py            # LLM provider abstraction (OpenAI/Anthropic)
│   │   │   ├── prompts.py              # Prompt templates
│   │   │   └── validators.py           # Output validation + confidence scoring
│   │   ├── graph/
│   │   │   ├── authority_analyzer.py    # Authority clustering + ranking
│   │   │   ├── citation_resolver.py     # Citation string → opinion ID resolution
│   │   │   └── conflict_detector.py    # Circuit split detection
│   │   ├── ingestion/
│   │   │   ├── bulk_loader.py           # CSV bulk ingestion
│   │   │   ├── chunker.py              # Legal-aware document chunking
│   │   │   ├── courtlistener.py        # CourtListener API client
│   │   │   └── parser.py              # Opinion text parsing + cleaning
│   │   ├── queue/
│   │   │   └── worker.py               # Redis-based async job processing
│   │   └── search/
│   │       ├── embeddings.py            # Embedding service (batch + cache)
│   │       └── vector_search.py        # Qdrant search with MMR reranking
│   ├── db/
│   │   ├── migrations/                  # Alembic migrations
│   │   ├── repositories/
│   │   │   ├── citation_repo.py         # Citation graph queries (recursive CTEs)
│   │   │   ├── extraction_repo.py       # Extraction CRUD
│   │   │   ├── job_repo.py             # Job tracking CRUD
│   │   │   └── opinion_repo.py         # Opinion CRUD
│   │   └── session.py                  # Async session factory
│   ├── utils/
│   │   ├── citation_parser.py           # Regex-based legal citation parser
│   │   └── text_cleaning.py            # HTML stripping, normalization
│   └── main.py                          # Uvicorn entry point
├── tests/
│   ├── conftest.py                      # Shared fixtures (db session, app, client)
│   ├── unit/                            # Isolated tests (mocked dependencies)
│   └── integration/                     # End-to-end tests (real Postgres + Qdrant)
├── infrastructure/
│   ├── Dockerfile                       # Multi-stage (base → deps → dev → prod)
│   ├── docker-compose.yml               # Local dev (API + Postgres + Qdrant + Redis)
│   └── terraform/
│       └── main.tf                      # GCP Cloud Run deployment
├── .github/workflows/ci.yml            # Lint → Type check → Test
├── pyproject.toml                       # uv + ruff + mypy + pytest config
├── alembic.ini                          # Database migration config
└── .env.example                         # Environment variable template
```

---

## How It Works

### 1. Ingestion

The ingestion pipeline fetches raw court opinions from the CourtListener API. Each opinion is parsed to strip HTML artifacts, normalized via `text_cleaning.py`, and split into semantically meaningful chunks by `chunker.py` — a legal-aware chunker that preserves citation boundaries, detects section headings (roman numerals, lettered subsections), and keeps footnotes attached to their parent paragraphs. Opinions and their metadata (court, date filed, jurisdiction) are persisted to PostgreSQL via the repository pattern.

### 2. Extraction

The extraction engine submits opinions to LLM providers (OpenAI GPT-4o or Anthropic Claude) via a provider-agnostic client abstraction. Each extraction request uses a carefully constructed prompt template that instructs the LLM to return structured JSON matching a Pydantic schema. The raw LLM output is **never trusted** — it passes through multi-stage validation:

- Pydantic schema validation (correct types, required fields)
- Business rule validation (disposition must be a known enum, confidence bounds)
- Confidence scoring (cross-referencing extracted data against source text)
- Retry with exponential backoff on transient failures

Extraction jobs run asynchronously via a Redis-backed queue with status tracking.

### 3. Citation Graph

The graph engine operates on extracted citation data:

- **Resolution**: Citation strings (e.g., "554 U.S. 570") are parsed via regex, matched to opinions in the database by volume/reporter/page, with fuzzy fallback by case name and approximate date
- **Subgraph construction**: Recursive CTEs traverse the citation graph outward and inward from a root opinion to a configurable depth
- **Authority ranking**: Opinions are scored by citation count, court level (SCOTUS > appellate > district), and recency
- **Conflict detection**: Pairs of opinions from different circuits are checked for overlapping cited authorities, opposing dispositions (affirmed vs. reversed), and conflicting citation types (follows vs. distinguishes). A weighted confidence score is computed for each potential circuit split

### 4. Search

Extracted holdings are embedded via OpenAI `text-embedding-3-large` (1536 dimensions) with an in-memory LRU cache and indexed in Qdrant. Search supports two strategies:

- **Similarity**: Standard cosine similarity search
- **MMR (Maximal Marginal Relevance)**: Fetches 50 candidates, then greedily selects results that maximize `λ * relevance - (1-λ) * redundancy`. Pairwise cosine similarity is computed via numpy for O(k·n) performance. This avoids near-duplicate holdings from opinions in the same circuit on the same issue.

---

## Design Decisions

### Why no LangChain?

LangChain adds ~50 transitive dependencies, leaky abstractions over simple SDK calls, and opaque prompt formatting. This project uses direct `openai` and `anthropic` SDK clients — every prompt is a visible string template, every retry is an explicit `tenacity` decorator, every validation step is a Pydantic model. When the LLM provider changes an API, we update one file, not debug three layers of framework magic. This is the approach described in "[I Reverse-Engineered 200 AI Startups](https://medium.com/)" — build on primitives, not on wrappers.

### Why MMR over simple similarity search?

Legal search has a specific failure mode: when you search for "qualified immunity," the top 10 results by cosine similarity are often 10 opinions from the same circuit saying essentially the same thing. MMR reranking with λ=0.7 keeps the most relevant result first, then penalizes subsequent results proportional to their similarity to already-selected results. The result is a diverse set spanning multiple circuits, time periods, and sub-issues — far more useful for legal research. The hand-rolled implementation uses numpy for pairwise cosine similarity, avoiding any dependency on a reranking library.

### Why legal-aware chunking?

Generic text splitters (split every N tokens) break citations mid-reference ("See 554 U.S." / "570 (2008)"), sever footnotes from their context, and split reasoning chains across chunks. The legal-aware chunker detects section boundaries (roman numerals, Part I/II/III patterns), preserves complete citation strings, keeps footnotes with their parent paragraphs, and respects natural paragraph boundaries. This costs ~200 lines of regex over a one-line `text_splitter.split()` call, but the extraction quality difference is measurable.

### Why hand-rolled extraction validation?

LLM outputs are probabilistic. A model might return `"disposition": "affirm"` instead of `"affirmed"`, hallucinate a citation that doesn't exist, or assign 0.99 confidence to a clearly wrong holding. The validation layer:

1. Normalizes enum values with fuzzy matching (levenshtein distance on disposition strings)
2. Cross-references extracted citations against the citation parser to verify format validity
3. Computes independent confidence scores by checking holding length, keyword density, and citation count against statistical baselines
4. Flags extractions below the confidence threshold for human review

This is the difference between a demo and a data pipeline. Every LLM output is an untrusted input that must be validated before it enters the database.
