# Real-Estate Vector Search — photo-aware listing search

Type one sentence, get homes — ranked by **meaning**, matched down to the
**exact photo**, with budgets that actually hold (`under $100k` means
under $100k). Or **drop a photo** and find listings that look like it.

Built and run end to end on real data: **7,300+ apartments, 60,000 photos,
millisecond answers on a second-hand RX 5700 XT**. One solo developer,
one cheap GPU, zero paid AI services.

| AI sentence search | Search by photo |
|---|---|
| ![AI search](docs/ai-search.png) | ![Photo search](docs/photo-search.png) |

Every result card shows the photo the engine actually matched — not the
cover shot — with a score. On the property page, matched photos carry
percentage ribbons: per-photo attribution, not black-box vibes.

| Property page |
|---|
| ![Property page](docs/property-page.png) |

## One-command demo

```bash
sudo ./run.sh   # starts Postgres + GPU embed server if down, then the site
# → http://127.0.0.1:8002
```

The `run.sh` here is machine-agnostic env-based; set `SUDO_PASSWORD` or
 MySQL-like cluster names to match your box, or run the three pieces manually:

```bash
sudo systemctl start postgresql            # Postgres 14+ with pgvector + unaccent
./embed-server.sh                          # GPU embedding server on :8081
../.venv/bin/python site.py                # site on :8002
```

First search is slow (model warm-up), then ~150ms — warm photo search ~200ms.

## What's inside

- **AI search** (`/api/aisearch`): deterministic constraints (price/rooms/area)
  become SQL pre-filters → EmbeddingGemma-2 vectors rank inside → RRF fusion
  across text/photo/joint signals → shortfalls backfill as *labeled*
  near-misses ("$130,000 — $10,000 over budget"). Never empty, never lying.
- **Search by photo** (`/api/imagesearch`): raw bytes in → 960px resize →
  image embedding with the media marker → exact scan across all three vector
  kinds → pool-normalized match %. Verified: feed a listing its own photo and
  it comes back top at ~99% in ~185ms warm.
- **Property pages** (`/property/<id>`): gallery + lightbox, specs table,
  price/m², mortgage estimate, amenities, similar homes (nearest *joint*
  property vectors, not price-band neighbors).
- **Classic filters** underneath: RESO-style ListPrice, BedroomsTotal,
  LivingArea, YearBuilt, City — AI ranking on top.

## How it works (30 seconds)

1. **Once, offline:** every description + every photo → 256-dim vectors
   (EmbeddingGemma-2, Apache 2.0) → pgvector (`listings` / `chunks` tables,
   HNSW m=16 / ef=64). Three chunk kinds per listing: text, per-image, joint
   property (text + all photos in one forward pass).
2. **Per search:** numbers extracted from the sentence become database rules
   first (vectors can't do inequalities); vectors rank survivors by meaning;
   RRF fuses the three signals; the winner's matching photo is cited.

Hybrid strategy (per-photo + joint vectors) solves a real problem: single
property-level vectors dilute photo signals (a pool in 1 of 40 photos) and
lose photo attribution. Joint vectors carry gestalt; per-photo vectors carry
recall and citation. RRF gets both.

### Retrieval design decisions worth stealing

- **Numbers first, vibes second.** "Under $100k" is parsed deterministically
  and enforced as SQL before any ranking — semantic similarity scores between
  inside-range and outside-range listings differ by ~0.02, so vectors alone
  will happily lie about budgets.
- **Strict-first with labeled backfill.** When nothing fits every rule,
  near-misses return clearly labeled ("over budget by $10k") instead of
  quietly breaking the user's constraint or returning an empty page.
- **Bilingual/multilingual for free.** EmbeddingGemma-2 is multilingual;
  documents are translated once (RU→EN) and search works in either language.
- **dHash near-dupe removal** (`hamming ≤ 5`) + blank-tile removal at prep
  time; 567 junk images dropped without quality loss.

## Repo layout

| Path | What |
|---|---|
| `real-estate-to-vec.py` | full pipeline: download → translate → prep → embed → load → verify |
| `site.py` + `site-static/` | demo website (FastAPI + vanilla JS) |
| `embed-server.sh` | GPU embedding server launcher (:8081, llama.cpp) |
| `run.sh` | one-command demo launcher (env-driven) |
| `docs/` | screenshots |

### Pipeline details

```bash
../.venv/bin/python real-estate-to-vec.py --step all --dataset bishkek
# stages run separately: download translate prep embed load verify
# embed needs ./embed-server.sh up (GPU, ~0.1-0.3 s/image → ~60k images in
# hours; the same job is ~23 s/image on CPU → weeks)
```

- `download`: HuggingFace dataset CSV + per-listing image trees (resumable,
  batched, loud failure if a run fetches zero files)
- `translate`: free Google endpoint, cached + resumable JSON
- `prep`: ALL images per listing in gallery order, 960px long side, dHash
  dedupe, EN docs, `manifest.jsonl`
- `embed`: 1 text + N image + 1 joint property vector per listing, 256-dim
  MRL (server returns 768 → client truncates + renormalizes); chunks that
  exceed the context budget get text + image vectors only (no degraded
  joint fallback, by design)
- `load`: `listings` + `chunks(kind='listing'|'image'|'property')`,
  `vector(256)`, HNSW index
- `verify`: fused mini-eval (RRF k=60), hit@1/hit@3 report

### Llama.cpp embedding server integration notes

Hard-won details that will save you days (see `embed-server.sh`):

- the media marker is **random per server start** — read it from `GET /props`
- task prefixes are REQUIRED or cross-modal ranking silently flips:
  `task: search result | query: ...` vs `title: ... | text: ...`
- multimodal responses nest as `[{"index", "embedding": [[...]]}]`
- multimodal forward passes must fit ONE ubatch — batch the server with
  `-ub 8192`, not the default 1024 (under-load joint calls 500 otherwise)
- server returns 768-dim for this model; 256-dim MRL truncation happens
  client-side + renormalize

## Dataset note

The demo dataset is Bishkek apartments: `raimbekovm/bishkek-real-estate`
(CC-BY-4.0). Photos belong to their owners. The code is dataset-agnostic —
it takes many numeric/text columns (`Price $…, N rooms, …` docs) and any
image folder tree, and there's a Flanders (Belgium) pipeline in the same
script via `--dataset flanders`.

## Requirements

Linux + Python 3.12+ venv (`sentence-transformers`, `torch`, `psycopg`,
`pandas`, `pillow`, `huggingface_hub`), Postgres 14+ with `pgvector` +
`unaccent`, llama.cpp built with Vulkan/CUDA → **AMD/NVIDIA GPU optional**
(CPU fallback: `--backend st`, painfully slow at image scale).

## License

Code: MIT. See `LICENSE`.
