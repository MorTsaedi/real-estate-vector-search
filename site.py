#!/usr/bin/env python3
"""
Vector Search Demo — real-estate website over the photo corpus.

Landing: AI demo search + normal filter search, categories, featured homes.
Property pages with galleries. All data from Postgres fl_* + data/prep photos.

Run:  ../.venv/bin/python site.py        -> http://127.0.0.1:8002
Needs: Postgres fl_listings/fl_chunks (real-estate-to-vec.py --step load),
       embedding server on :8081 for /api/aisearch (works without it otherwise).
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import time
import urllib.request
from functools import lru_cache

import psycopg
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

HERE = pathlib.Path(__file__).parent
STATIC = HERE / "site-static"
PREP = HERE / "data" / "prep"

DSN = (
    f"host={os.environ.get('PGHOST', 'localhost')} "
    f"port={os.environ.get('PGPORT', '5432')} "
    f"dbname={os.environ.get('PGDATABASE', 'rag')} "
    f"user={os.environ.get('PGUSER', 'postgres')} "
    f"password={os.environ.get('PGPASSWORD', 'postgres')}"
)
EMBED_URL = os.environ.get("LLAMA_EMBED_URL",
                           "http://127.0.0.1:8081/embedding")
EMBED_PROPS = "http://127.0.0.1:8081/props"
Q_PREFIX = "task: search result | query: "

app = FastAPI(title="Vector Search Demo")
app.mount("/img", StaticFiles(directory=PREP), name="img")
app.mount("/s", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/property/{listing_id}")
def property_page(listing_id: str):
    return FileResponse(STATIC / "property.html")


def _conn():
    return psycopg.connect(DSN)


@lru_cache(maxsize=None)
def _photos_of(lid: str) -> tuple[str, ...]:
    """Gallery URLs in order. Cover = first. Cached: files only change when
    we clean the corpus ourselves (re-run once after corpus edits)."""
    files = sorted(PREP.glob(f"{lid}_*.jpg"))
    return tuple(f"/img/{p.name}" for p in files)


# Russian spec values → English (closed vocabularies in this corpus).
RU_EN = {
    "евроремонт": "European-style renovation", "среднее": "average",
    "не достроено": "unfinished", "под самоотделку (псо)": "needs finishing",
    "хорошее": "good",
    "монолит": "monolithic", "панель": "panel", "кирпич": "brick",
    "без отопления": "no heating", "смешанное": "mixed",
    "центральное": "central", "на твердом топливе": "solid fuel",
    "автономное": "autonomous", "на газе": "gas", "электрическое": "electric",
    "2 с/у и более": "2+ bathrooms", "нет": "none",
    "раздельный": "separate", "совмещенный": "combined",
    "частично меблирована": "partly furnished", "пустая": "unfurnished",
    "полностью меблирована": "fully furnished",
    "плитка": "tile", "ламинат": "laminate", "линолеум": "linoleum",
    "дерево": "wood", "паркет": "parquet", "пробковое": "cork",
    "балкон": "balcony", "застекленный балкон": "glazed balcony",
    "лоджия": "loggia",
    "паркинг": "parking garage", "рядом охраняемая стоянка": "guarded lot nearby",
    "гараж": "garage",
    "домофон": "intercom", "сигнализация": "alarm", "охрана": "guard",
    "видеонаблюдение": "CCTV", "видеодомофон": "video intercom",
    "кодовый замок": "keypad lock", "консьерж": "concierge",
    "решетки на окнах": "window bars",
    "оптика": "fiber", "adsl": "ADSL", "проводной": "wired",
    "от собственника": "from owner", "от застройщика": "from developer",
    "от агента": "from agent",
    "агентство": "agency", "риелтор": "agent",
}


def _en_spec(value, split_parts: bool = False):
    """Translate a RU spec value (or comma-list) to English."""
    if not isinstance(value, str) or not value.strip():
        return None
    if split_parts:
        parts = [RU_EN.get(p.strip(), p.strip())
                 for p in value.split(",") if p.strip()]
        return ", ".join(p for p in parts if p) or None
    return RU_EN.get(value.strip(), value.strip())


# Match-%: normalized per query over the whole retrieved pool (all three
# kinds' shortlists), mapped to 5..99. Rank-preserving and honest about what
# it is: how close this hit is versus everything the query considered.
# Absolute cosine distances in this corpus bunch up (~0.18-0.30 for every
# top hit), so any absolute formula prints a near-constant number.


def _card(lid: str, r: dict, n_photos: int) -> dict:
    price = r.get("price")
    pf = float(price) if price is not None else None
    area = r.get("area_m2")
    ppm = None
    if r.get("price_per_m2") is not None:
        ppm = float(r["price_per_m2"])
    elif pf and area:
        try:
            ppm = round(pf / float(area))
        except (TypeError, ValueError, ZeroDivisionError):
            ppm = None
    fl, ft = r.get("floor_n"), r.get("floors_total")
    floor = None
    try:
        if fl is not None:
            floor = f"{int(float(fl))}" + (f"/{int(float(ft))}" if ft is not None else "")
    except (TypeError, ValueError):
        floor = None
    return {
        "listing_id": lid,
        "price": pf,
        "price_per_m2": ppm,
        "bedrooms": r.get("bedrooms"),
        "area_m2": area,
        "floor": floor,
        "district": r.get("municipality") or "Bishkek",
        "cover": (_photos_of(lid)[0] if _photos_of(lid) else None),
        "n_photos": n_photos,
        "snippet": (r.get("text") or "")[:160],
    }


def _mortgage(price: float | None) -> int | None:
    """Rough monthly payment: 20% down, 18% annual, 20 years. Estimate only."""
    if not price:
        return None
    principal = price * 0.8
    r = 0.18 / 12
    n = 240
    return round(principal * r / (1 - (1 + r) ** -n))


_CARD_COLS = ("listing_id, price, bedrooms, area_m2, municipality, text,"
              " floor_n, floors_total, price_per_m2")


def _vecstr(a) -> str:
    if isinstance(a, str):  # psycopg returns vector as "[...]" text
        s = a.strip()
        return s if s.startswith("[") else f"[{s}]"
    return "[" + ",".join(f"{float(x):.6f}" for x in a) + "]"


def _photo_counts(cur, lids: list[str]) -> dict[str, int]:
    if not lids:
        return {}
    cur.execute("SELECT listing_id, count(*) FROM fl_chunks "
                "WHERE kind = 'image' AND listing_id = ANY(%s) "
                "GROUP BY 1", [lids])
    return {lid: n for lid, n in cur.fetchall()}


@app.get("/api/stats")
def api_stats():
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM fl_listings")
            n_list = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM fl_chunks WHERE kind='image'")
            n_photos = cur.fetchone()[0]
            cur.execute("SELECT bedrooms::int, count(*) FROM fl_listings "
                        "GROUP BY 1 ORDER BY 1")
            rows = [{"rooms": b, "n": n} for b, n in cur.fetchall()]
            rooms = [r for r in rows if r["rooms"] in (1, 2, 3)]
            n4 = sum(r["n"] for r in rows if (r["rooms"] or 0) >= 4)
            rooms.append({"rooms": 4, "plus": True, "n": n4})
            cur.execute(
                "SELECT 'Under $100k', count(*) FROM fl_listings WHERE price < 100000 "
                "UNION ALL SELECT '$100–200k', count(*) FROM fl_listings "
                "WHERE price >= 100000 AND price < 200000 "
                "UNION ALL SELECT '$200k+', count(*) FROM fl_listings WHERE price >= 200000")
            bands = [{"label": lb, "n": n} for lb, n in cur.fetchall()]
            feats = []
            for label, cond in [
                    ("Air conditioning", "text ILIKE '%air conditioner%'"),
                    ("Quiet courtyard", "text ILIKE '%quiet courtyard%'"),
                    ("Built-in kitchen", "text ILIKE '%built-in kitchen%'"),
                    ("Parking", "text ILIKE '%parking%'")]:
                cur.execute(f"SELECT count(*) FROM fl_listings WHERE {cond}")
                feats.append({"label": label, "n": cur.fetchone()[0],
                              "q": label.lower()})
            cur.execute("SELECT municipality, count(*) FROM fl_listings "
                        "WHERE municipality <> '' AND municipality <> 'Bishkek' "
                        "GROUP BY 1 ORDER BY 2 DESC LIMIT 15")
            districts = [{"name": d, "n": n} for d, n in cur.fetchall()]
            cur.execute("SELECT count(*) FROM fl_listings WHERE year_built >= 2020")
            newbuild = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM fl_listings WHERE area_m2 >= 100")
            spacious = cur.fetchone()[0]
    return {"listings": n_list, "photos": n_photos,
            "rooms": rooms, "bands": bands, "features": feats,
            "districts": districts,
            "standards": [{"label": "New construction", "n": newbuild,
                           "q": "new", "hint": "YearBuilt ≥ 2020"},
                          {"label": "Spacious 100 m²+", "n": spacious,
                           "q": "big", "hint": "LivingArea ≥ 100"}]}


@app.get("/api/featured")
def api_featured(n: int = 8):
    n = max(1, min(n, 24))
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT listing_id, price, bedrooms, area_m2, municipality, text, floor_n, floors_total, price_per_m2
                   FROM fl_listings ORDER BY random() LIMIT %s""", [n])
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            counts = _photo_counts(cur, [r["listing_id"] for r in rows])
    return [_card(r["listing_id"], r, counts.get(r["listing_id"], 0))
            for r in rows]


@app.get("/api/browse")
def api_browse(rooms: str = "", rooms_min: str = "",
               min_price: str = "", max_price: str = "",
               min_area: str = "", max_area: str = "",
               min_year: str = "",
               amenity: str = "", district: str = "",
               sort: str = "price_asc", limit: int = 12):
    # RESO Data Dictionary mapping: ListPrice, BedroomsTotal, LivingArea,
    # YearBuilt, City. (This corpus: all Residential apartments, Bishkek.)
    limit = max(1, min(limit, 48))
    sorts = {"price_asc": "price ASC NULLS LAST",
             "price_desc": "price DESC NULLS LAST",
             "rooms": "bedrooms DESC NULLS LAST",
             "area": "area_m2 DESC NULLS LAST",
             "year": "year_built DESC NULLS LAST"}
    order = sorts.get(sort, sorts["price_asc"])
    where, params = [], []
    try:
        if rooms.strip():
            where.append("bedrooms = %s")
            params.append(int(rooms))
        if rooms_min.strip():
            where.append("bedrooms >= %s")
            params.append(int(rooms_min))
        if min_price.strip():
            where.append("price >= %s")
            params.append(float(min_price))
        if max_price.strip():
            where.append("price <= %s")
            params.append(float(max_price))
        if min_area.strip():
            where.append("area_m2 >= %s")
            params.append(float(min_area))
        if max_area.strip():
            where.append("area_m2 <= %s")
            params.append(float(max_area))
        if min_year.strip():
            where.append("year_built >= %s")
            params.append(int(min_year))
    except ValueError:
        raise HTTPException(status_code=400, detail="bad numeric filter")
    if amenity.strip():
        where.append("text ILIKE %s")
        params.append(f"%{amenity.strip()}%")
    if district.strip():
        where.append("municipality ILIKE %s")
        params.append(f"%{district.strip()}%")
    w = ("WHERE " + " AND ".join(where)) if where else ""
    t0 = time.perf_counter()
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM fl_listings {w}", params)
            matched = cur.fetchone()[0]
            cur.execute(
                f"""SELECT listing_id, price, bedrooms, area_m2, municipality, text, floor_n, floors_total, price_per_m2
                    FROM fl_listings {w} ORDER BY {order} LIMIT {limit}""",
                params)
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            counts = _photo_counts(cur, [r["listing_id"] for r in rows])
    return {"matched": matched, "sort": sort,
            "ms": round((time.perf_counter() - t0) * 1000, 1),
            "results": [_card(r["listing_id"], r, counts.get(r["listing_id"], 0))
                        for r in rows]}


# ---------------------------------------------------------------- constraints
# Deterministic extraction: numbers are database rules, never vector vibes.
# Hard (strict SQL): price, rooms, area — numeric columns, exact.
# Soft (displayed, ranked semantically): amenities, bathroom wishes.
_PRICE_TOK = re.compile(r"\$?\s*([\d,]+(?:\.\d+)?)\s*([kKmM]?)\b")
_AREA_TOK = re.compile(r"([\d,]+)\s*(m2|m²|sqm|sq\s*m)\b", re.I)
_UNDER = re.compile(r"(under|below|less than|cheaper than|max|up to|within|not over|not more than)\b", re.I)
_OVER = re.compile(r"(over|above|more than|min\b|at least|starting at|from)\b", re.I)
_ROOMS = re.compile(r"(\d+)\+?[\s-]*(?:bedrooms?|beds?|br\b|rooms?|r\b)", re.I)
_BATHS = re.compile(r"(\d+(?:\.\d+)?)\s*(?:bathroom|bathrooms|bath|baths|ba\b)", re.I)
_KNOWN_AMEN = ["air conditioner", "quiet courtyard", "built-in kitchen",
               "parking", "open-plan kitchen", "separate rooms",
               "not a corner unit", "new plumbing", "pantry",
               "upgraded finish", "suitable for business", "balcony",
               "furnished", "renovated", "parking space"]


def _money(dollar: str, num: str, suffix: str):
    try:
        v = float(num.replace(",", ""))
    except ValueError:
        return None
    if suffix.lower() == "k":
        v *= 1000
    elif suffix.lower() == "m":
        v *= 1_000_000
    elif not dollar and v < 10000:
        return None  # bare small number ("2 rooms") is not a price
    return v


def extract_constraints(q: str) -> dict:
    """-> {hard: {max_price?, min_price?, min_rooms?, min_area?, max_area?},
            soft: [amenity phrases, bath wishes]}. Never raises."""
    hard: dict = {}
    soft: list[str] = []
    # --- price tokens (skip ones that belong to an area match) ---
    area_spans = [m.span() for m in _AREA_TOK.finditer(q)]
    toks = []
    for m in _PRICE_TOK.finditer(q):
        if any(s <= m.start() < e for s, e in area_spans):
            continue
        v = _money(m.group(0).strip()[:1] == "$", m.group(1), m.group(2))
        if v is not None:
            toks.append((v, m.start(), m.end()))
    if len(toks) >= 2:
        gap = q[toks[0][2]:toks[1][1]]
        if re.search(r"between", q[:toks[0][1]], re.I) and re.search(r"\band\b", gap, re.I) \
                or re.match(r"^\s*[-–]\s*$", gap) or re.match(r"^\s+to\s+$", gap, re.I):
            hard["min_price"] = min(toks[0][0], toks[1][0])
            hard["max_price"] = max(toks[0][0], toks[1][0])
    if "min_price" not in hard:
        for v, s, _ in toks:
            before = q[:s][-32:]
            if _UNDER.search(before):
                hard.setdefault("max_price", v)
            elif _OVER.search(before):
                hard.setdefault("min_price", v)
    # --- rooms / area / baths ---
    m = _ROOMS.search(q)
    if m:
        hard["min_rooms"] = int(m.group(1))
    for m in _AREA_TOK.finditer(q):
        try:
            v = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        before = q[:m.start()][-32:]
        if _UNDER.search(before):
            hard["max_area"] = v if "max_area" not in hard else min(hard["max_area"], v)
        elif _OVER.search(before):
            hard["min_area"] = v if "min_area" not in hard else max(hard["min_area"], v)
        else:
            hard["min_area"] = v if "min_area" not in hard else max(hard["min_area"], v)
    m = _BATHS.search(q)
    if m:
        soft.append(f"{m.group(1)} bathroom(s)")
    low = q.lower()
    soft.extend(a for a in _KNOWN_AMEN if a in low)
    return {"hard": hard, "soft": soft}


def _violations(row: dict, hard: dict) -> list[str]:
    """Human-readable list of which hard rules a row breaks (for badges)."""
    out = []
    try:
        p = float(row["price"]) if row["price"] is not None else None
        if hard.get("max_price") is not None and p is not None and p > hard["max_price"]:
            out.append(f"${p:,.0f} over ${hard['max_price']:,.0f} budget")
        if hard.get("min_price") is not None and p is not None and p < hard["min_price"]:
            out.append(f"${p:,.0f} under ${hard['min_price']:,.0f} minimum")
        b = row["bedrooms"]
        b = int(float(b)) if b is not None else None
        if hard.get("min_rooms") is not None and b is not None and b < hard["min_rooms"]:
            out.append(f"{b} rooms, wanted {hard['min_rooms']}+")
        a = row["area_m2"]
        a = float(a) if a is not None else None
        if hard.get("min_area") is not None and a is not None and a < hard["min_area"]:
            out.append(f"{a:g} m², wanted {hard['min_area']:g}+")
        if hard.get("max_area") is not None and a is not None and a > hard["max_area"]:
            out.append(f"{a:g} m², over {hard['max_area']:g} max")
    except (TypeError, ValueError):
        pass
    return out


def _embed_query(q: str) -> list[float]:
    try:
        with urllib.request.urlopen(EMBED_PROPS, timeout=10) as r:
            json.load(r)  # server must be up; marker not needed for text
    except Exception as e:
        raise HTTPException(status_code=503,
                            detail=f"embedding server down (:8081): {e}")
    payload = {"content": [Q_PREFIX + q]}
    req = urllib.request.Request(EMBED_URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            body = json.load(r)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"embed failed: {e}")
    v = body[0]["embedding"] if isinstance(body, list) else body["embedding"]
    while isinstance(v[0], list):
        v = v[0]
    import numpy as np
    a = np.asarray(v, dtype=np.float32)[:256]
    return (a / (float((a ** 2).sum() ** 0.5) or 1)).tolist()


@app.get("/api/aisearch")
def api_aisearch(q: str, k: int = 6, offset: int = 0):
    # Option 3 (strict-first hybrid): hard numeric rules are SQL pre-filters
    # (vectors can't do inequalities); semantic rank runs INSIDE the set.
    # Shortfalls backfill with labeled near-misses — never empty, never lying.
    # Pagination: combined order (strict first, then backfill) sliced by offset.
    k = max(1, min(k, 24))
    offset = max(0, offset)
    t0 = time.perf_counter()
    cons = extract_constraints(q)
    hard = cons["hard"]
    where, params = [], []
    if hard.get("max_price") is not None:
        where.append("price <= %s")
        params.append(hard["max_price"])
    if hard.get("min_price") is not None:
        where.append("price >= %s")
        params.append(hard["min_price"])
    if hard.get("min_rooms") is not None:
        where.append("bedrooms >= %s")
        params.append(hard["min_rooms"])
    if hard.get("min_area") is not None:
        where.append("area_m2 >= %s")
        params.append(hard["min_area"])
    if hard.get("max_area") is not None:
        where.append("area_m2 <= %s")
        params.append(hard["max_area"])
    w = (" AND " + " AND ".join(where)) if where else ""
    qv = _embed_query(q)
    vs = "[" + ",".join(f"{x:.6f}" for x in qv) + "]"

    def ranked(cur, extra_where: str, extra_params: list,
               exclude: list, limit: int):
        per: dict[str, list[str]] = {}
        best_photo: dict[str, int] = {}
        min_dist: dict[str, float] = {}
        if extra_where:
            # HNSW post-filters under-return on selective WHEREs; the
            # filtered set is small, so an exact scan is both correct and fast.
            cur.execute("SET LOCAL enable_indexscan = off")
        for kind in ("listing", "image", "property"):
            cur.execute(
                f"""SELECT c.listing_id, c.text,
                           c.embedding <=> '{vs}'::vector AS d FROM fl_chunks c
                    JOIN fl_listings l USING (listing_id)
                    WHERE c.kind = '{kind}' {extra_where}
                    ORDER BY d
                    LIMIT {limit}""", extra_params)
            rows = cur.fetchall()
            per[kind] = [lid for lid, _, _ in rows]
            for lid, txt, d in rows:
                dd = float(d)
                if lid not in min_dist or dd < min_dist[lid]:
                    min_dist[lid] = dd
            if kind == "image":
                for lid, txt, _ in rows:
                    # chunk text is "Photo i/n" (1-based); files are 0-based
                    m = re.match(r"Photo (\d+)/", txt or "")
                    if lid not in best_photo and m:
                        best_photo[lid] = max(0, int(m.group(1)) - 1)
        scores: dict[str, float] = {}
        for kind, lids in per.items():
            for rk, lid in enumerate(lids):
                if lid in exclude:
                    continue
                scores[lid] = scores.get(lid, 0.0) + 1.0 / (60 + rk)
        # RRF selects the candidate set; final order is closest-meaning first.
        order = sorted(scores, key=lambda lid: min_dist.get(lid, 9.0))
        return order, best_photo, min_dist

    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SET hnsw.ef_search = 64")
            strict_full, best_photo, min_dist = ranked(cur, w, params, [], 200)
            backfill_full: list[str] = []
            if w:
                more, bp2, md2 = ranked(cur, "", [], strict_full, 200)
                backfill_full = more
                best_photo.update(bp2)
                min_dist.update(md2)
            combined = strict_full + backfill_full
            total = len(combined)
            page = combined[offset:offset + k]
            strict_page = [lid for lid in page if lid in set(strict_full)]
            backfill_page = [lid for lid in page if lid not in set(strict_full)]
            want = page
            if not want:
                return {"results": [], "backfill": [], "filters": cons,
                        "ms": 0, "total": total, "offset": offset}
            cur.execute(
                """SELECT listing_id, price, bedrooms, area_m2, municipality, text,
                          floor_n, floors_total, price_per_m2
                   FROM fl_listings WHERE listing_id = ANY(%s)""", [want])
            cols = [d[0] for d in cur.description]
            by_id = {r[0]: dict(zip(cols, r)) for r in cur.fetchall()}
            counts = _photo_counts(cur, want)
    ms = (time.perf_counter() - t0) * 1000
    pool = [d for d in min_dist.values()]
    pmin, pmax = (min(pool), max(pool)) if pool else (0.0, 1.0)

    def card_for(lid: str) -> dict:
        r = by_id.get(lid)
        if not r:
            return {}
        c = _card(lid, r, counts.get(lid, 0))
        idx = best_photo.get(lid, 0)
        c["match_photo"] = f"/img/{lid}_{idx:03d}.jpg"
        c["match_idx"] = idx
        c["violations"] = _violations(r, hard)
        d = min_dist.get(lid)
        if d is not None:
            c["distance"] = round(d, 4)
            c["match_pct"] = 99 if pmax <= pmin else \
                max(5, min(99, round(5 + 94 * (pmax - d) / (pmax - pmin))))
        return c

    return {"results": [c for c in (card_for(lid) for lid in strict_page) if c],
            "backfill": [c for c in (card_for(lid) for lid in backfill_page) if c],
            "filters": cons, "ms": round(ms, 1),
            "total": total, "offset": offset}


# ---------------------------------------------------------------- image search
# Upload a photo of the home you want; we embed it with the SAME model that
# vectorized all ~60k listing photos, then rank homes by distance in that
# shared space. Exact scan (75k vectors) - small corpus, exact is both
# honest and fast here.

MAX_UPLOAD = 12_000_000


def _embed_image_jpeg(data: bytes) -> str:
    """Image bytes -> base64 JPEG (long side 960, like the corpus) ready for
    the multimodal embed call. Raises HTTPException on bad input."""
    import base64
    import io
    try:
        from PIL import Image
    except ImportError:
        raise HTTPException(status_code=500, detail="Pillow missing on server")
    try:
        im = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="not a readable image")
    w, h = im.size
    scale = 960 / max(w, h)  # corpus photos are 960px long side; stay in-distribution
    if scale < 1:
        im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                       Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def _embed_query_image(data: bytes) -> list[float]:
    b64 = _embed_image_jpeg(data)
    try:
        with urllib.request.urlopen(EMBED_PROPS, timeout=10) as r:
            marker = json.load(r)["media_marker"]
    except Exception as e:
        raise HTTPException(status_code=503,
                            detail=f"embedding server down (:8081): {e}")
    payload = {"content": [{"prompt_string": marker, "multimodal_data": [b64]}]}
    req = urllib.request.Request(EMBED_URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            body = json.load(r)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"image embed failed: {e}")
    v = body[0]["embedding"] if isinstance(body, list) else body["embedding"]
    while isinstance(v[0], list):
        v = v[0]
    import numpy as np
    a = np.asarray(v, dtype=np.float32)[:256]
    return (a / (float((a ** 2).sum() ** 0.5) or 1)).tolist()


@app.post("/api/imagesearch")
async def api_imagesearch(request: Request, k: int = 12):
    """Search by photo: raw image bytes in the body (no multipart needed).
    Ranks homes by their closest vector of ANY kind (per-photo, joint
    property, description) - the ✨ badge shows the matched photo."""
    data = await request.body()
    if len(data) < 100:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(data) > MAX_UPLOAD:
        raise HTTPException(status_code=413, detail="image too large (12 MB max)")
    k = max(1, min(k, 24))
    t0 = time.perf_counter()
    qv = _embed_query_image(data)
    vs = "[" + ",".join(f"{x:.6f}" for x in qv) + "]"

    best_photo: dict[str, int] = {}
    min_dist: dict[str, float] = {}
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL enable_indexscan = off")  # exact scan
            cur.execute(
                f"""SELECT listing_id, kind, text,
                           embedding <=> '{vs}'::vector AS d
                    FROM fl_chunks
                    WHERE kind IN ('image', 'property', 'listing')
                    ORDER BY d
                    LIMIT 1600""")
            for lid, kind, txt, d in cur.fetchall():
                dd = float(d)
                if lid not in min_dist or dd < min_dist[lid]:
                    min_dist[lid] = dd
                if kind == "image" and lid not in best_photo:
                    m = re.match(r"Photo (\d+)/", txt or "")
                    if m:
                        best_photo[lid] = max(0, int(m.group(1)) - 1)
            order = sorted(min_dist, key=min_dist.get)[:k]
            if not order:
                return {"results": [], "ms": round((time.perf_counter() - t0) * 1000, 1)}
            cur.execute(
                """SELECT listing_id, price, bedrooms, area_m2, municipality, text,
                          floor_n, floors_total, price_per_m2
                   FROM fl_listings WHERE listing_id = ANY(%s)""", [order])
            cols = [d[0] for d in cur.description]
            by_id = {r[0]: dict(zip(cols, r)) for r in cur.fetchall()}
            counts = _photo_counts(cur, order)
    ms = (time.perf_counter() - t0) * 1000
    pool = list(min_dist.values())
    pmin, pmax = (min(pool), max(pool)) if pool else (0.0, 1.0)

    def card_for(lid: str) -> dict:
        r = by_id.get(lid)
        if not r:
            return {}
        c = _card(lid, r, counts.get(lid, 0))
        idx = best_photo.get(lid, 0)
        c["match_photo"] = f"/img/{lid}_{idx:03d}.jpg"
        c["match_idx"] = idx
        d = min_dist.get(lid)
        if d is not None:
            c["distance"] = round(d, 4)
            c["match_pct"] = 99 if pmax <= pmin else \
                max(5, min(99, round(5 + 94 * (pmax - d) / (pmax - pmin))))
        return c

    return {"results": [c for c in (card_for(lid) for lid in order) if c],
            "ms": round(ms, 1)}


@app.get("/api/photo_scores")
def api_photo_scores(listing_id: str, q: str):
    """Per-photo match % for one listing vs a query. Normalized within the
    home (best photo = highest); rank-preserving, same ordering as search."""
    qv = _embed_query(q)
    vs = "[" + ",".join(f"{x:.6f}" for x in qv) + "]"
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""SELECT text, embedding <=> '{vs}'::vector AS d
                    FROM fl_chunks
                    WHERE listing_id = %s AND kind = 'image'
                    ORDER BY d""", [listing_id])
            rows = cur.fetchall()
    if not rows:
        return {"scores": []}
    ds = [float(d) for _, d in rows]
    lo, hi = min(ds), max(ds)
    out = []
    for txt, d in rows:
        m = re.match(r"Photo (\d+)/", txt or "")
        idx = max(0, int(m.group(1)) - 1) if m else 0
        pct = 99 if hi <= lo else max(5, min(99, round(5 + 94 * (hi - d) / (hi - lo))))
        out.append({"idx": idx, "pct": pct, "distance": round(float(d), 4)})
    return {"scores": sorted(out, key=lambda s: s["idx"])}


@app.get("/api/property/{listing_id}")
def api_property(listing_id: str):
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT listing_id, price, bedrooms, area_m2, property_type,
                          municipality, text, text_src, living_area_m2,
                          kitchen_area_m2, floor_n, floors_total, ceiling_h,
                          building_type, year_built, condition, heating,
                          bathroom, balcony, parking, furniture, flooring,
                          security, internet, address, price_per_m2,
                          seller_type, latitude, longitude
                   FROM fl_listings WHERE listing_id = %s""", [listing_id])
            row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="unknown listing")
            cols = [d[0] for d in cur.description]
            r = dict(zip(cols, row))
    photos = _photos_of(listing_id)
    similar = []
    with _conn() as conn:
        with conn.cursor() as cur:
            # Closest PROPERTY vectors to this property's own joint vector
            # (falls back to its listing vector when it has no joint one).
            cur.execute(
                """SELECT embedding FROM fl_chunks
                   WHERE listing_id = %s AND kind = 'property' LIMIT 1""",
                [listing_id])
            anchor = cur.fetchone()
            if anchor is None:
                cur.execute(
                    """SELECT embedding FROM fl_chunks
                       WHERE listing_id = %s AND kind = 'listing' LIMIT 1""",
                    [listing_id])
                anchor = cur.fetchone()
            if anchor is not None:
                cur.execute("SET hnsw.ef_search = 64")
                cur.execute(
                    f"""SELECT listing_id FROM fl_chunks
                        WHERE kind = 'property' AND listing_id != %s
                        ORDER BY embedding <=> '{_vecstr(anchor[0])}'::vector
                        LIMIT 8""", [listing_id])
                sim_ids = [x[0] for x in cur.fetchall()]
                if sim_ids:
                    cur.execute(
                        f"""SELECT {_CARD_COLS}
                            FROM fl_listings WHERE listing_id = ANY(%s)""",
                        [sim_ids])
                    cols = [d[0] for d in cur.description]
                    rows = [dict(zip(cols, x)) for x in cur.fetchall()]
                    counts = _photo_counts(cur, sim_ids)
                    by_id = {x["listing_id"]: x for x in rows}
                    similar = [_card(lid, by_id[lid], counts.get(lid, 0))
                               for lid in sim_ids if lid in by_id]
    price = r["price"]
    pf = float(price) if price is not None else None
    area = r["area_m2"]
    try:
        af = float(area) if area is not None else None
    except (TypeError, ValueError):
        af = None
    ppm = float(r["price_per_m2"]) if r.get("price_per_m2") is not None else \
        (round(pf / af) if pf and af else None)
    text = r["text"] or ""
    known = ["quiet courtyard", "plastic windows", "separate rooms",
             "not a corner unit", "built-in kitchen", "new plumbing",
             "air conditioner", "pantry", "upgraded finish",
             "open-plan kitchen", "suitable for business"]
    low = text.lower()
    amenities = [a for a in known if a in low]

    def num(x):
        try:
            f = float(x)
            return int(f) if f.is_integer() else f
        except (TypeError, ValueError):
            return None

    def txt(x):
        v = (x or "").strip() if isinstance(x, str) else x
        return v or None

    fl, ft = r.get("floor_n"), r.get("floors_total")
    try:
        floor = (f"{int(float(fl))} of {int(float(ft))}"
                 if fl is not None and ft is not None
                 else (f"{int(float(fl))}" if fl is not None else None))
    except (TypeError, ValueError):
        floor = None
    specs = [
        ("Rooms", num(r["bedrooms"])),
        ("Total area", f"{num(area)} m²" if num(area) is not None else None),
        ("Living area", f"{num(r['living_area_m2'])} m²"
         if num(r.get("living_area_m2")) is not None else None),
        ("Kitchen area", f"{num(r['kitchen_area_m2'])} m²"
         if num(r.get("kitchen_area_m2")) is not None else None),
        ("Floor", floor),
        ("Condition", _en_spec(r.get("condition"))),
        ("Building type", _en_spec(r.get("building_type"))),
        ("Year built", num(r.get("year_built"))),
        ("Ceiling height", f"{num(r['ceiling_h'])} m"
         if num(r.get("ceiling_h")) is not None else None),
        ("Heating", _en_spec(r.get("heating"))),
        ("Bathroom", _en_spec(r.get("bathroom"))),
        ("Balcony", _en_spec(r.get("balcony"))),
        ("Parking", _en_spec(r.get("parking"))),
        ("Furniture", _en_spec(r.get("furniture"))),
        ("Flooring", _en_spec(r.get("flooring"))),
        ("Security", _en_spec(r.get("security"), split_parts=True)),
        ("Internet", _en_spec(r.get("internet"))),
    ]
    specs = [{"k": k, "v": str(v)} for k, v in specs if v is not None]
    address = txt(r.get("address"))
    return {
        "listing_id": listing_id,
        "price": pf,
        "price_per_m2": ppm,
        "mortgage": _mortgage(pf),
        "bedrooms": num(r["bedrooms"]), "area_m2": num(area),
        "district": r["municipality"] or "Bishkek",
        "address": address,
        "seller": _en_spec(r.get("seller_type")) or txt(r.get("seller_type")),
        "description": r["text"], "description_src": r.get("text_src") or "",
        "amenities": amenities, "specs": specs,
        "photos": photos, "n_photos": len(photos), "similar": similar,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
