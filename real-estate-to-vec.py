#!/usr/bin/env python3
"""
Real-estate -> vectors (text + images) -> pgvector.

Datasets (--dataset):
  bishkek  (default, OPEN): raimbekovm/bishkek-real-estate — 8.8k Bishkek
           apartments, RU descriptions, ~64k photos in images/<id>/ folders.
  flanders (GATED): AnthonyL1996/immovlan-flanders — 5.7k listings, 116k
           photos, Dutch descriptions (access: hf auth login + request form).

  1. Log in:  hf auth login                         # needs a huggingface.co account
  2. Open https://huggingface.co/datasets/AnthonyL1996/immovlan-flanders
     and click "Request access" (name, affiliation, institutional email,
     intended use; tick both checkboxes). Approval is manual - can take days.
  3. Re-run this script; the download stage will then work.

Pipeline (run all, or one --step at a time):
  download : snapshot the CSVs + image tars into ./data/raw/
  translate: descriptions -> English via free Google endpoint, resumable
             cache ./data/translations.json (--src-lang nl|ru|…)
  prep     : extract ALL images per listing, resize (long side 960px),
             build ENGLISH text docs -> ./data/prep/manifest.jsonl
             (one row per listing, with an ordered "images" list)
  embed    : EmbeddingGemma 2 (text+vision, 256-dim MRL) -> ./data/vec/*.npz
             (1 text vector + N image vectors + 1 JOINT property vector
             [text + all photos, one call] per listing)
  load     : create fl_listings / fl_chunks in Postgres, HNSW index
  verify   : smoke-test cross-modal retrieval (Dutch text query vs images)

Run:  ../.venv/bin/python real-estate-to-vec.py --step all --max-listings 200
      ../.venv/bin/python real-estate-to-vec.py --step all            # full set

Embed stage needs: pip install "sentence-transformers>=6.1.0" torch pillow transformers
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import tarfile

REPO = "AnthonyL1996/immovlan-flanders"
BISHKEK_REPO = "raimbekovm/bishkek-real-estate"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RAW = os.path.join(DATA, "raw")
RAW_BISHKEK = os.path.join(DATA, "raw-bishkek")
PREP = os.path.join(DATA, "prep")
VEC = os.path.join(DATA, "vec")

CSV_PATTERNS = [
    "tabular_data/listings.csv",
    "tabular_data/descriptions.csv",
    "tabular_data/splits_grouped.csv",
    "data/processed/listings_clean.csv",
]
FULLRES_PATTERNS = ["data/processed/photos_full_res/*.tar"]
S224_PATTERNS = ["data/processed/shards_224_v2/*.tar",
                 "data/processed/shards_224_v2/*/*.tar"]

EG2_MODEL = os.environ.get("EG2_MODEL", "google/embeddinggemma-2")
EG2_DIM = int(os.environ.get("EG2_DIM", "256"))  # 128/256/512/768 (MRL)
# GPU embedding server (./embed-server.sh): text+vision on RX 5700 XT.
LLAMA_EMBED_URL = os.environ.get("LLAMA_EMBED_URL",
                                 "http://127.0.0.1:8081/embedding")
Q_PREFIX = "task: search result | query: "


def _build_title(row: dict) -> str:
    """Short English title improves document embeddings (per EG2 docs: a
    real title beats 'none'). Falls back gracefully when fields are missing."""
    ccy = row.get("ccy", "€")
    bits = [row.get("property_type") or "home",
            f"in {row['municipality']}" if row.get("municipality") else "",
            f"{ccy}{row['price']}" if row.get("price") else "",
            f"{row['bedrooms']} bedrooms" if row.get("bedrooms") else "",
            f"{row['area_m2']} m²" if row.get("area_m2") else ""]
    return " ".join(b for b in bits if b) or row.get("listing_id", "")


def _doc_prompt(title: str, text: str) -> str:
    return f"title: {title or 'none'} | text: {_clip_text(text)}"


def _clip_text(text: str, limit: int = 7000) -> str:
    """Keep embedding inputs under the 8K context (EG2 counts ~1 tok/4 chars
    for Latin scripts). DB keeps the full text; only the vector sees this."""
    return text if len(text) <= limit else text[:limit]


def _norm_listing_id(lid: str) -> str:
    """Strip URL-suffix dupes (Bishkek has '<id>?cadastre-details-block')."""
    return (lid or "").split("?")[0].strip()


def _numstr(v) -> str:
    """'3.0'->'3', '90.8'->'90.8', ''->'' — CSV numbers read as floats."""
    s = str(v or "").strip()
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return s


def _dhash(raw: bytes) -> int | None:
    """64-bit difference hash, dependency-free. None if undecodable."""
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(raw)).convert("L").resize((9, 8))
        px = list(im.getdata())
    except Exception:
        return None
    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | (px[y * 9 + x] > px[y * 9 + x + 1])
    return bits


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _is_blank(raw: bytes) -> bool:
    """Drop single-color placeholders (white 'no photo' tiles, etc.)."""
    try:
        from PIL import Image
        import statistics
        im = Image.open(io.BytesIO(raw)).convert("L").resize((32, 32))
        return statistics.pstdev(im.getdata()) < 2.0
    except Exception:
        return True  # undecodable -> treat as junk


def _clean_gallery(items: list[bytes]) -> tuple[list[bytes], dict]:
    """Drop junk + near-dupes (hamming <= 5), keep gallery order."""
    stats = {"blank": 0, "dupe": 0}
    seen: list[int] = []
    out = []
    for raw in items:
        if _is_blank(raw):
            stats["blank"] += 1
            continue
        h = _dhash(raw)
        if h is not None and any(_hamming(h, s) <= 5 for s in seen):
            stats["dupe"] += 1
            continue
        if h is not None:
            seen.append(h)
        out.append(raw)
    return out, stats

ACCESS_HELP = """
ERROR: cannot download - HTTP 401/403. The dataset is gated (manual approval).

  1. ../.venv/bin/python -m huggingface_hub.commands.huggingface_cli login
     (or: hf auth login) with a Hugging Face token
  2. Request access at https://huggingface.co/datasets/AnthonyL1996/immovlan-flanders
     ("Request access": name + affiliation + institutional email + intended use,
     tick both checkboxes). Educational/research use fits their terms.
  3. Wait for approval, then re-run this step.
""".strip()


# ------------------------------------------------------------ translate
TRANS_CACHE = os.path.join(DATA, "translations.json")
TRANS_DELIM = "\n\n§§§\n\n"
TRANS_MAX_CHARS = 4800  # Google single-request budget (limit ~5000)


def _save_trans_cache(cache: dict) -> None:
    tmp = TRANS_CACHE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, ensure_ascii=False)
    os.replace(tmp, TRANS_CACHE)


def _load_trans_cache() -> dict:
    if not os.path.exists(TRANS_CACHE):
        return {}
    try:
        with open(TRANS_CACHE, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, UnicodeDecodeError):
        print(f"WARNING: {TRANS_CACHE} corrupt; starting fresh")
        return {}


class _Translator:
    """Batched src->EN via the free Google endpoint (no key, no download).
    Falls back to smaller/one-by-one requests on misalignment."""

    def __init__(self, src_lang: str):
        import requests
        self.s = requests.Session()
        self.src = src_lang

    def _one(self, text: str, attempt: int = 0):
        import time
        import requests
        try:
            r = self.s.get(
                "https://translate.googleapis.com/translate_a/single",
                params={"client": "gtx", "sl": self.src, "tl": "en",
                        "dt": "t", "q": text},
                timeout=45)
        except requests.RequestException:
            if attempt < 4:
                time.sleep(2 ** attempt)
                return self._one(text, attempt + 1)
            return None
        if r.status_code in (429, 503) and attempt < 4:
            time.sleep(5 * (attempt + 1))
            return self._one(text, attempt + 1)
        if r.status_code != 200:
            return None
        try:
            return "".join(s[0] for s in r.json()[0] if s and s[0])
        except (ValueError, TypeError, IndexError):
            return None

    def translate(self, items: list[str]) -> list:
        import random
        import time
        out: list = [None] * len(items)
        batches, cur, size = [], [], 0
        for gi, it in enumerate(items):
            if cur and size + len(it) > TRANS_MAX_CHARS:
                batches.append(cur)
                cur, size = [], 0
            cur.append((gi, it))
            size += len(it) + len(TRANS_DELIM)
        if cur:
            batches.append(cur)
        for bi, batch in enumerate(batches):
            joined = TRANS_DELIM.join(t for _, t in batch)
            res = self._one(joined)
            parts = res.split("§§§") if res else []
            if len(parts) == len(batch):
                for (gi, _), part in zip(batch, parts):
                    out[gi] = part.strip()
            else:
                for gi, text in batch:
                    out[gi] = self._one(text) or text
            if bi % 25 == 0:
                time.sleep(0.05 + random.random() * 0.25)
        return out


def step_translate(src_lang: str, csv_path: str, id_col: str,
                   text_col: str) -> None:
    try:
        import requests  # noqa: F401
    except ImportError:
        die("need: ../.venv/bin/pip install requests")
    _, rows = _read_csv(csv_path)
    texts = [(r.get(id_col, ""), (r.get(text_col) or "").strip())
             for r in rows]
    texts = [(i, t) for i, t in texts if t]
    print(f"descriptions found: {len(texts)}")
    cache = _load_trans_cache()
    pending = [(i, t) for i, t in texts if t not in cache]
    print(f"translating {len(pending)} ({src_lang}->en), "
          f"{len(texts) - len(pending)} cached ...")
    if pending:
        import time
        tr = _Translator(src_lang)
        t0 = time.time()
        for k in range(0, len(pending), 40):
            chunk = pending[k:k + 40]
            for (_, src), out in zip(chunk,
                                     tr.translate([t for _, t in chunk])):
                if out:
                    cache[src] = out
            _save_trans_cache(cache)
            el = time.time() - t0
            rate = (k + len(chunk)) / el if el else 0
            print(f"  {k + len(chunk)}/{len(pending)} ({rate:.1f}/s)",
                  flush=True)
    missing = sum(1 for _, t in texts if t not in cache)
    print(f"translate done: cache {TRANS_CACHE} "
          f"({len(cache)} entries, {missing} untranslated)")


def die(msg: str) -> None:
    print(f"\nERROR: {msg}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------- download
def _pick_bishkek_ids(max_listings: int) -> list[str]:
    """First N base ids (CSV order) with description + downloaded photos."""
    _, rows = _read_csv(os.path.join(RAW_BISHKEK, "listings.csv"))
    seen: set[str] = set()
    ids = []
    for r in rows:
        lid = _norm_listing_id(r.get("listing_id", ""))
        if not lid or lid in seen:
            continue
        seen.add(lid)
        if (r.get("description") or "").strip() and \
                str(r.get("photos_downloaded", "0")) not in ("0", "", "nan"):
            ids.append(lid)
        if 0 < max_listings <= len(ids):
            break
    return ids


def step_download(image_set: str, dataset: str, max_listings: int) -> None:
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import GatedRepoError
    except ImportError:
        die("need huggingface_hub: ../.venv/bin/pip install huggingface_hub")
    if dataset == "bishkek":
        return _step_download_bishkek(max_listings, snapshot_download,
                                      GatedRepoError)
    patterns = list(CSV_PATTERNS)
    patterns += FULLRES_PATTERNS if image_set == "full" else S224_PATTERNS
    try:
        path = snapshot_download(REPO, repo_type="dataset",
                                 allow_patterns=patterns,
                                 local_dir=RAW, local_dir_use_symlinks=False,
                                 max_workers=4)
    except GatedRepoError:
        die(ACCESS_HELP)
    except Exception as e:
        if "401" in str(e) or "403" in str(e):
            die(ACCESS_HELP)
        raise
    print(f"downloaded -> {path}")
    total = 0
    for dp, _, fns in os.walk(RAW):
        for fn in fns:
            total += os.path.getsize(os.path.join(dp, fn))
    print(f"raw size: {total / 1e9:.1f} GB")


# ---------------------------------------------------------------- prep
def _read_csv(path: str) -> tuple[list[str], list[dict]]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return rows[0].keys() if rows else [], rows


def _resize_hero(img_bytes: bytes, long_side: int = 960) -> bytes:
    from PIL import Image
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    im.thumbnail((long_side, long_side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def _step_download_bishkek(max_listings: int, snapshot_download,
                           GatedRepoError) -> None:
    pats = ["listings.csv"]
    try:
        snapshot_download(BISHKEK_REPO, repo_type="dataset",
                          allow_patterns=pats,
                          local_dir=RAW_BISHKEK, local_dir_use_symlinks=False,
                          max_workers=12)
    except GatedRepoError:
        die("unexpected gate on the open Bishkek repo?!")
    ids = _pick_bishkek_ids(max_listings)
    print(f"listings with description+photos: {len(ids)} "
          f"(~{len(ids) * 8.2 * 142 / 1e6:.1f} GB photos est.)")
    # NOTE: ~7k allow_patterns in ONE snapshot_download silently match
    # nothing (huggingface_hub 1.33.0) — batch them.
    BATCH = 400
    for b in range(0, len(ids), BATCH):
        group = ids[b:b + BATCH]
        pats = [f"images/{lid}/*" for lid in group]
        if b == 0:
            pats.append("listings.csv")
        snapshot_download(BISHKEK_REPO, repo_type="dataset",
                          allow_patterns=pats,
                          local_dir=RAW_BISHKEK,
                          local_dir_use_symlinks=False, max_workers=12)
        print(f"  batch {b // BATCH + 1}/{(len(ids) + BATCH - 1) // BATCH} "
              f"({min(b + BATCH, len(ids))}/{len(ids)} listings)", flush=True)
    got = sum(len([f for f in fs if f.lower().endswith(".jpg")])
              for _, _, fs in os.walk(os.path.join(RAW_BISHKEK, "images")))
    if ids and got == 0:
        die("photo download fetched 0 files — pattern matching failed, aborting")
    path = RAW_BISHKEK
    total = 0
    for dp, _, fns in os.walk(RAW_BISHKEK):
        for fn in fns:
            if "/.cache/" not in dp:
                total += os.path.getsize(os.path.join(dp, fn))
    print(f"downloaded -> {path} ({total / 1e9:.1f} GB, resumable: re-run)")


def step_prep(image_set: str, max_listings: int, images_per_listing: int,
              dataset: str) -> None:
    if dataset == "bishkek":
        return _step_prep_bishkek(max_listings, images_per_listing)
    from PIL import Image  # noqa: F401  (fail fast if pillow missing)
    _, listings = _read_csv(os.path.join(RAW, "tabular_data/listings.csv"))
    _, descs = _read_csv(os.path.join(RAW, "tabular_data/descriptions.csv"))
    # Normalize ids everywhere (?suffix dupes) and drop duplicate rows.
    seen_rows: set[str] = set()
    clean_listings = []
    for r in listings:
        lid = _norm_listing_id(r.get("listing_id", ""))
        if not lid or lid in seen_rows:
            continue
        seen_rows.add(lid)
        r["listing_id"] = lid
        clean_listings.append(r)
    listings = clean_listings
    desc_by_id = {_norm_listing_id(d.get("listing_id", "")):
                  d.get("description_clean", "") for d in descs}
    split_by_id: dict[str, str] = {}
    sp = os.path.join(RAW, "tabular_data/splits_grouped.csv")
    if os.path.exists(sp):
        for r in _read_csv(sp)[1]:
            split_by_id[_norm_listing_id(r.get("listing_id", ""))] = \
                r.get("split", "")

    with_photos = [r for r in listings if (r.get("photo_paths") or "").strip()]
    print(f"listings: {len(listings)}, with photos: {len(with_photos)}")
    if max_listings > 0:
        with_photos = with_photos[:max_listings]
        print(f"subset: first {len(with_photos)} listings with photos")

    os.makedirs(PREP, exist_ok=True)

    # listing_id -> ordered image bytes (gallery order for full-res).
    gallery: dict[str, list[bytes]] = {r["listing_id"]: [] for r in with_photos}
    if image_set == "full":
        # photo_paths basenames are gallery order and match full-res tar
        # member names <listing_id>/<basename>.
        want: dict[str, list[str]] = {}
        for r in with_photos:
            paths = [os.path.basename(p.strip())
                     for p in r["photo_paths"].split(",") if p.strip()]
            if images_per_listing > 0:
                paths = paths[:images_per_listing]
            want[r["listing_id"]] = paths
        index = {(lid, base): i for lid, bases in want.items()
                 for i, base in enumerate(bases)}
        res_dir = os.path.join(RAW, "data/processed/photos_full_res")
        tars = [os.path.join(res_dir, f) for f in sorted(os.listdir(res_dir))
                if f.endswith(".tar")]
        for tar_path in tars:
            with tarfile.open(tar_path) as tf:
                for m in tf:
                    if not m.isfile():
                        continue
                    parts = m.name.split("/")
                    if len(parts) != 2:
                        continue
                    key = (parts[0], parts[1])
                    if key in index:
                        gallery[parts[0]].append(
                            (index[key], tf.extractfile(m).read()))
            print(f"  collected {tar_path}", flush=True)
        for lid in gallery:
            gallery[lid].sort(key=lambda t: t[0])
            gallery[lid] = [b for _, b in gallery[lid]]
    else:
        # 224px shards: {listing_id}_{idx}.jpg. NOTE idx is alphabetical
        # rank, NOT gallery position, so order here is not gallery order.
        want_ids = set(gallery)
        found: dict[str, list[tuple[int, str]]] = {lid: [] for lid in want_ids}
        sh_dir = os.path.join(RAW, "data/processed/shards_224_v2")
        for dp, _, fns in os.walk(sh_dir):
            for fn in sorted(fns):
                if not fn.endswith(".jpg"):
                    continue
                stem = fn[:-len(".jpg")]
                lid, _, idx = stem.rpartition("_")
                if lid in want_ids and idx.isdigit():
                    found[lid].append((int(idx), os.path.join(dp, fn)))
        for lid, items in found.items():
            items.sort()
            if images_per_listing > 0:
                items = items[:images_per_listing]
            gallery[lid] = []
            for _, path in items:
                with open(path, "rb") as fh:
                    gallery[lid].append(fh.read())
        print("NOTE: s224 order is alphabetical rank, NOT gallery order. "
              "Use --image-set full for true gallery order.")

    n_noimg = 0
    n_img = 0
    n_blank_dupe = {"blank": 0, "dupe": 0}
    man_path = os.path.join(PREP, "manifest.jsonl")
    tr_cache = _load_trans_cache()
    if tr_cache:
        print(f"  using translations cache ({len(tr_cache)} entries)")
    else:
        print("  no translations cache — docs stay in source language "
              "(run --step translate first)")
    with open(man_path, "w", encoding="utf-8") as out:
        for r in with_photos:
            lid = r["listing_id"]
            imgs = gallery.get(lid, [])
            if not imgs:
                n_noimg += 1
                continue
            imgs, st = _clean_gallery(imgs)
            n_blank_dupe["blank"] += st["blank"]
            n_blank_dupe["dupe"] += st["dupe"]
            if not imgs:
                n_noimg += 1
                continue
            paths = []
            for i, raw in enumerate(imgs):
                ip = os.path.join(PREP, f"{lid}_{i:03d}.jpg")
                with open(ip, "wb") as fh:
                    fh.write(_resize_hero(raw) if image_set == "full" else raw)
                paths.append(ip)
            n_img += len(paths)
            desc = desc_by_id.get(lid, "")
            en = (tr_cache.get(desc, "") or "").strip() or desc
            # English doc: translated description + key structured fields.
            doc = (f"{en} Price €{r.get('asking_price', '')}, "
                   f"{r.get('n_bedrooms', '')} bedrooms, "
                   f"{r.get('surface_m2', '')} m², {r.get('property_type', '')}, "
                   f"postcode {r.get('postcode', '')}.")
            src_doc = (f"{desc} Prijs €{r.get('asking_price', '')}, "
                       f"{r.get('n_bedrooms', '')} slaapkamers, "
                       f"{r.get('surface_m2', '')} m², {r.get('property_type', '')}, "
                       f"postcode {r.get('postcode', '')}.")
            row = {"listing_id": lid, "split": split_by_id.get(lid, ""),
                   "price": r.get("asking_price", ""),
                   "bedrooms": r.get("n_bedrooms", ""),
                   "area_m2": r.get("surface_m2", ""),
                   "property_type": r.get("property_type", ""),
                   "postcode": r.get("postcode", ""),
                   "municipality": r.get("municipality", "")}
            out.write(json.dumps({
                **row, "title": _build_title({**row, "listing_id": lid}),
                "text": doc, "text_src": src_doc, "images": paths,
            }, ensure_ascii=False) + "\n")
    print(f"prep done: {n_img} images across "
          f"{len(with_photos) - n_noimg} listings, "
          f"{n_noimg} listings w/o image skipped, "
          f"junk dropped: {n_blank_dupe} -> {man_path}")


def _step_prep_bishkek(max_listings: int, images_per_listing: int) -> None:
    """Bishkek layout: images/<id>/NN.jpg (numeric = gallery order).
    Reads data/listings_en.csv (English, preferred) or falls back to
    raw listings.csv + translations cache. Original RU kept as text_src
    when available."""
    from PIL import Image  # noqa: F401
    en_path = os.path.join(DATA, "listings_en.csv")
    if os.path.exists(en_path):
        _, rows = _read_csv(en_path)
        text_col, src_col = "description_en", None
        print(f"  using English CSV: {en_path}")
    else:
        _, rows = _read_csv(os.path.join(RAW_BISHKEK, "listings.csv"))
        text_col, src_col = "description", "description"
    seen: set[str] = set()
    clean = []
    for r in rows:
        lid = _norm_listing_id(r.get("listing_id", ""))
        if not lid or lid in seen:
            continue
        seen.add(lid)
        r["listing_id"] = lid
        clean.append(r)
    img_root = os.path.join(RAW_BISHKEK, "images")
    tr_cache = _load_trans_cache()
    if tr_cache:
        print(f"  using translations cache ({len(tr_cache)} entries)")
    elif src_col:
        print("  no translations cache — run "
              "--step translate --src-lang ru first!")
    rev_cache = {v: k for k, v in tr_cache.items()} if tr_cache else {}
    if max_listings > 0:
        clean = clean[:max_listings]
    os.makedirs(PREP, exist_ok=True)
    n_noimg = n_img = 0
    junk = {"blank": 0, "dupe": 0}
    man_path = os.path.join(PREP, "manifest.jsonl")
    with open(man_path, "w", encoding="utf-8") as out:
        for r in clean:
            lid = r["listing_id"]
            en = (r.get(text_col) or "").strip()
            if src_col:  # raw CSV path: translate via cache
                en = (tr_cache.get(en, "") or "").strip() or en
                ru = (r.get(src_col) or "").strip()
            else:  # English CSV: recover original RU from cache
                ru = rev_cache.get(en, "")
            desc = en
            ldir = os.path.join(img_root, lid)
            files = []
            if os.path.isdir(ldir):
                files = sorted(f for f in os.listdir(ldir)
                               if f.lower().endswith(".jpg"))
            if not desc or not files:
                n_noimg += 1
                continue
            if images_per_listing > 0:
                files = files[:images_per_listing]
            raw = []
            for fn in files:
                with open(os.path.join(ldir, fn), "rb") as fh:
                    raw.append(fh.read())
            raw, st = _clean_gallery(raw)
            junk["blank"] += st["blank"]
            junk["dupe"] += st["dupe"]
            if not raw:
                n_noimg += 1
                continue
            paths = []
            for i, b in enumerate(raw):
                ip = os.path.join(PREP, f"{lid}_{i:03d}.jpg")
                with open(ip, "wb") as fh:
                    fh.write(_resize_hero(b))
                paths.append(ip)
            n_img += len(paths)
            amen_raw = (r.get("amenities_en") or "").strip()
            if not amen_raw and r.get("amenities"):
                # raw-CSV fallback: map RU phrases through the cache
                amen_raw = "; ".join(
                    tr_cache.get(p.strip(), p.strip())
                    for p in r["amenities"].split(",") if p.strip())
            amen = amen_raw
            price = _numstr(r.get("price_usd", ""))
            rooms = _numstr(r.get("rooms", ""))
            area = _numstr(r.get("total_area_m2", ""))
            district = (r.get("district") or "").strip()
            doc = (f"{en} Price ${price}, {rooms} rooms, {area} m², "
                   f"apartment in {district}, Bishkek."
                   + (f" Amenities: {amen}." if amen else ""))
            src_doc = f"{ru} Цена ${price}, {rooms} комнат, {area} м²." \
                if ru else ""
            row = {"listing_id": lid, "split": "", "price": price,
                   "bedrooms": rooms, "area_m2": area,
                   "property_type": "apartment", "postcode": "",
                   "municipality": district or "Bishkek", "ccy": "$"}
            out.write(json.dumps({
                **row, "title": _build_title({**row, "listing_id": lid}),
                "text": doc, "text_src": src_doc, "images": paths,
            }, ensure_ascii=False) + "\n")
    print(f"prep done: {n_img} images, {n_noimg} skipped (no desc/photos), "
          f"junk dropped: {junk} -> {man_path}")


# ---------------------------------------------------------------- embed
def _load_manifest() -> list[dict]:
    with open(os.path.join(PREP, "manifest.jsonl"), encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _mrl_256(vecs, dim: int):
    """Truncate 768-dim MRL vectors to dim + re-normalize (unit length)."""
    import numpy as np
    out = []
    for v in vecs:
        v = np.asarray(v, dtype=np.float32)[:dim]
        n = float(np.linalg.norm(v)) or 1.0
        out.append(v / n)
    return out


def _llama_post(content, timeout: int = 300):
    import json
    import urllib.request
    req = urllib.request.Request(
        LLAMA_EMBED_URL, data=json.dumps({"content": content}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.load(resp)
    vecs = []
    items = body if isinstance(body, list) else [body]
    for it in items:
        v = it["embedding"] if isinstance(it, dict) else it
        vecs.append(v[0] if isinstance(v[0], list) else v)
    return vecs


def _llama_marker() -> str:
    import json
    import urllib.request
    with urllib.request.urlopen("http://127.0.0.1:8081/props",
                                timeout=10) as resp:
        return json.load(resp)["media_marker"]


def step_embed(dim: int, batch: int, backend: str) -> None:
    if dim not in (128, 256, 512, 768):
        die("--dim must be one of 128/256/512/768 (MRL)")
    man = _load_manifest()
    n_img = sum(len(m["images"]) for m in man)
    if backend == "llama":
        _step_embed_llama(man, n_img, dim, batch)
        return
    try:
        from sentence_transformers import SentenceTransformer
        from PIL import Image
        import numpy as np
    except ImportError:
        die('need: ../.venv/bin/pip install "sentence-transformers>=6.1.0" '
            "torch pillow transformers")
    print(f"embedding {len(man)} listings + {n_img} images "
          f"with {EG2_MODEL} @{dim}d (CPU/torch) ...")
    model = SentenceTransformer(EG2_MODEL,
                                config_kwargs={"audio_config": None})
    texts = [_doc_prompt(m.get("title", ""), m["text"]) for m in man]
    t_embs = model.encode(texts, batch_size=batch, show_progress_bar=True,
                          truncate_dim=dim, normalize_embeddings=True)
    # Flatten all images; keys are "<listing_id>::<photo_idx>".
    img_keys, imgs = [], []
    for m in man:
        for i, path in enumerate(m["images"]):
            img_keys.append(f"{m['listing_id']}::{i}")
            with Image.open(path) as im:
                imgs.append(im.convert("RGB").copy())
    i_embs = model.encode(imgs, batch_size=batch, show_progress_bar=True,
                          truncate_dim=dim, normalize_embeddings=True)
    # Joint property vectors: text + ALL photos in one call each.
    # Listings that don't fit at full budget get NO joint vector.
    j_keys, j_embs_in = [], []
    n_skipped = 0
    for m in man:
        doc = _doc_prompt(m.get("title", ""), m["text"])
        if not _joint_fits(len(m["images"]), doc):
            n_skipped += 1
            print(f"  {m['listing_id']}: too big for joint call "
                  f"({len(m['images'])} photos) — skipped", flush=True)
            continue
        pils = []
        for path in m["images"]:
            with Image.open(path) as im:
                pils.append(im.convert("RGB").copy())
        sample = {"text": _doc_prompt(m.get("title", ""), m["text"])
                  + " " + " ".join(["<|image|>"] * len(pils)),
                  "image": pils[0] if len(pils) == 1 else pils}
        j_keys.append(m["listing_id"])
        j_embs_in.append(sample)
    j_embs = model.encode(j_embs_in, batch_size=1, show_progress_bar=True,
                          truncate_dim=dim, normalize_embeddings=True)
    os.makedirs(VEC, exist_ok=True)
    import numpy as np  # noqa: F811
    np.savez_compressed(os.path.join(VEC, f"text_{dim}d.npz"),
                        **{m["listing_id"]: v for m, v in zip(man, t_embs)})
    np.savez_compressed(os.path.join(VEC, f"image_{dim}d.npz"),
                        **{k: v for k, v in zip(img_keys, i_embs)})
    np.savez_compressed(os.path.join(VEC, f"property_{dim}d.npz"),
                        **{k: v for k, v in zip(j_keys, j_embs)})
    print(f"saved -> {VEC}/text_{dim}d.npz ({len(man)} x {dim}d) + "
          f"image_{dim}d.npz ({len(img_keys)} x {dim}d) + "
          f"property_{dim}d.npz ({len(j_keys)} x {dim}d)")


def _step_embed_llama(man: list[dict], n_img: int, dim: int,
                      batch: int) -> None:
    """GPU path: EmbeddingGemma 2 GGUF via ./embed-server.sh (Vulkan).

    ~0.1s/image on RX 5700 XT (vs ~25s on CPU): ~116k photos in ~3-4h.
    Requires the same task prefixes as sentence-transformers, and MRL
    truncation client-side (server returns full 768-dim).
    """
    import base64
    import numpy as np
    try:
        marker = _llama_marker()
    except Exception:
        die("embedding server not reachable on :8081 - start ./embed-server.sh")
    print(f"embedding {len(man)} listings + {n_img} images via llama-server "
          f"(GPU) @{dim}d ...")
    t_vecs: dict[str, list] = {}
    for i in range(0, len(man), batch):
        chunk = man[i:i + batch]
        prompts = [_doc_prompt(m.get("title", ""), m["text"]) for m in chunk]
        try:
            vecs = _llama_post(prompts)
        except Exception as e:
            # Batch too long for ctx? Retry one by one (each is clipped).
            print(f"  text batch {i}-{i + len(chunk)} failed ({e}); "
                  f"retrying singles", flush=True)
            vecs = []
            for p in prompts:
                try:
                    vecs.extend(_llama_post([p]))
                except Exception as e2:
                    print(f"  SKIP text (server refused): {str(e2)[:100]}",
                          flush=True)
                    vecs.append(None)
        for m, v in zip(chunk, vecs):
            if v is not None:
                t_vecs[m["listing_id"]] = v
        print(f"  text {min(i + batch, len(man))}/{len(man)}", flush=True)
    os.makedirs(VEC, exist_ok=True)
    tv = {k: v for k, v in zip(t_vecs, _mrl_256(t_vecs.values(), dim))}
    np.savez_compressed(os.path.join(VEC, f"text_{dim}d.npz"), **tv)
    print(f"  text saved ({len(tv)} x {dim}d)", flush=True)
    i_vecs: dict[str, list] = {}
    done = 0
    for m in man:
        for j, path in enumerate(m["images"]):
            with open(path, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode()
            v = _llama_post([{"prompt_string": marker,
                              "multimodal_data": [b64]}])[0]
            i_vecs[f"{m['listing_id']}::{j}"] = v
            done += 1
            if done % 200 == 0:
                print(f"  images {done}/{n_img}", flush=True)
            if done % 2000 == 0:  # checkpoint: a crash keeps most work
                iv = {k: v for k, v in zip(i_vecs, _mrl_256(i_vecs.values(), dim))}
                np.savez_compressed(os.path.join(VEC, f"image_{dim}d.npz"), **iv)
                print(f"  images checkpoint ({len(iv)})", flush=True)
    iv = {k: v for k, v in zip(i_vecs, _mrl_256(i_vecs.values(), dim))}
    np.savez_compressed(os.path.join(VEC, f"image_{dim}d.npz"), **iv)
    print(f"  images saved ({len(iv)} x {dim}d)", flush=True)
    j_vecs: dict[str, list] = {}
    n_skipped = 0
    for k, m in enumerate(man):
        paths = m["images"]
        doc = _doc_prompt(m.get("title", ""), m["text"])
        if not _joint_fits(len(paths), doc):
            n_skipped += 1
            print(f"  {m['listing_id']}: too big for joint call "
                  f"({len(paths)} photos) — skipped", flush=True)
            continue
        b64s = []
        for path in paths:
            with open(path, "rb") as fh:
                b64s.append(base64.b64encode(fh.read()).decode())
        prompt = doc + " " + " ".join([marker] * len(b64s))
        try:
            v = _llama_post(
                [{"prompt_string": prompt, "multimodal_data": b64s}])[0]
        except Exception:
            import time as _t
            _t.sleep(5)  # transient hiccup? one retry, then skip
            try:
                v = _llama_post(
                    [{"prompt_string": prompt, "multimodal_data": b64s}])[0]
            except Exception as e2:
                n_skipped += 1
                print(f"  {m['listing_id']}: joint failed ({str(e2)[:80]})"
                      f" — skipped", flush=True)
                continue
        j_vecs[m["listing_id"]] = v
        if (k + 1) % 200 == 0:
            print(f"  joint {k + 1}/{len(man)}", flush=True)
        if (k + 1) % 500 == 0:
            jv = {k2: v2 for k2, v2 in
                  zip(j_vecs, _mrl_256(j_vecs.values(), dim))}
            np.savez_compressed(os.path.join(VEC, f"property_{dim}d.npz"), **jv)
    if n_skipped:
        print(f"  joint skipped total: {n_skipped} (text+photo vectors kept)")
    jv = {k: v for k, v in zip(j_vecs, _mrl_256(j_vecs.values(), dim))}
    np.savez_compressed(os.path.join(VEC, f"property_{dim}d.npz"), **jv)
    print(f"saved -> {VEC}/text_{dim}d.npz ({len(tv)} x {dim}d) + "
          f"image_{dim}d.npz ({len(iv)} x {dim}d) + "
          f"property_{dim}d.npz ({len(jv)} x {dim}d)")


# Joint property vector: text + ALL photos in ONE call. Simple rule: a
# listing either fits at full budget or gets NO joint vector (text +
# per-photo vectors are always kept). No degraded fallbacks.
JOINT_CTX = 8000
IMG_BUDGET = 280


def _joint_fits(n_all: int, text: str) -> bool:
    return len(text) // 4 + n_all * IMG_BUDGET <= JOINT_CTX


# ---------------------------------------------------------------- load
DSN = (
    f"host={os.environ.get('PGHOST', 'localhost')} "
    f"port={os.environ.get('PGPORT', '5432')} "
    f"dbname={os.environ.get('PGDATABASE', 'rag')} "
    f"user={os.environ.get('PGUSER', 'postgres')} "
    f"password={os.environ.get('PGPASSWORD', 'postgres')}"
)


def step_load(dim: int) -> None:
    try:
        import numpy as np
        import psycopg
    except ImportError:
        die("need: ../.venv/bin/pip install numpy psycopg")
    man = _load_manifest()
    t = np.load(os.path.join(VEC, f"text_{dim}d.npz"))
    v = np.load(os.path.join(VEC, f"image_{dim}d.npz"))
    p = np.load(os.path.join(VEC, f"property_{dim}d.npz"))

    def vecstr(a) -> str:
        return "[" + ",".join(f"{x:.6f}" for x in a) + "]"

    import psycopg
    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("DROP TABLE IF EXISTS fl_chunks")
            cur.execute("DROP TABLE IF EXISTS fl_listings")
            cur.execute(
                """CREATE TABLE fl_listings (
                     listing_id TEXT PRIMARY KEY, split TEXT, price NUMERIC,
                     bedrooms NUMERIC, area_m2 NUMERIC, property_type TEXT,
                     postcode TEXT, municipality TEXT, text TEXT,
                     text_src TEXT)""")
            cur.execute(
                f"""CREATE TABLE fl_chunks (
                     id SERIAL PRIMARY KEY, listing_id TEXT REFERENCES fl_listings,
                     kind TEXT, text TEXT, embedding vector({dim}))""")
            for m in man:
                lid = m["listing_id"]
                cur.execute(
                    "INSERT INTO fl_listings VALUES "
                    "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (lid, m["split"], m["price"] or None, m["bedrooms"] or None,
                     m["area_m2"] or None, m["property_type"], m["postcode"],
                     m["municipality"], m["text"], m.get("text_src", "")))
                if lid in t:  # server may refuse overlong texts
                    cur.execute(
                        f"INSERT INTO fl_chunks (listing_id, kind, text, embedding) "
                        f"VALUES (%s,'listing',%s,'{vecstr(t[lid])}'::vector)",
                        (lid, m["text"]))
                n = len(m["images"])
                for i in range(n):
                    cur.execute(
                        f"INSERT INTO fl_chunks (listing_id, kind, text, embedding) "
                        f"VALUES (%s,'image',%s,'{vecstr(v[f'{lid}::{i}'])}'::vector)",
                        (lid, f"Photo {i + 1}/{n} of {lid} "
                              f"({m['property_type']}, {m['municipality']})"))
                if lid in p:  # oversized listings have no joint vector
                    cur.execute(
                        f"INSERT INTO fl_chunks (listing_id, kind, text, embedding) "
                        f"VALUES (%s,'property',%s,'{vecstr(p[lid])}'::vector)",
                        (lid, f"{m['text']} [{n} photos]"))
            cur.execute("SELECT count(*) FROM fl_chunks")
            print(f"chunks loaded: {cur.fetchone()[0]}")
            print("building HNSW index (m=16, ef_construction=64) ...", flush=True)
            cur.execute("CREATE INDEX fl_chunks_hnsw ON fl_chunks "
                        "USING hnsw (embedding vector_cosine_ops) "
                        "WITH (m = 16, ef_construction = 64)")
        conn.commit()
    print("load done: fl_listings + fl_chunks ready")


# ---------------------------------------------------------------- verify
# Mini-eval: kind-aware retrieval. Each kind (listing/image/property) ranks
# on its own, then RRF fuses per listing. Flat mixed search lets the
# text-heavy joint vectors drown photo hits; fusion keeps every signal alive.
EVAL = [
    ("house with pool and big garden", "TEST-A"),
    ("apartment with modern kitchen", "TEST-B"),
    ("big garden next to the house", "TEST-A"),
]
RRF_K = 60
PER_KIND_K = 10


def _rrf_fuse(ranked_per_kind: dict[str, list[str]]) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for _kind, lids in ranked_per_kind.items():
        for rank, lid in enumerate(lids):
            scores[lid] = scores.get(lid, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def step_verify(dim: int, backend: str) -> None:
    import psycopg
    queries = [q for q, _ in EVAL]
    if backend == "llama":
        try:
            qvecs = _mrl_256(_llama_post([Q_PREFIX + q for q in queries]), dim)
        except Exception:
            die("embedding server not reachable - start ./embed-server.sh")
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EG2_MODEL,
                                    config_kwargs={"audio_config": None})
        qvecs = model.encode(queries, prompt_name="SearchQuery",
                             truncate_dim=dim, normalize_embeddings=True)
    hits1 = hits3 = 0
    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("SET hnsw.ef_search = 64")
            for i, (q, want) in enumerate(EVAL):
                qv = qvecs[i]
                vs = "[" + ",".join(f"{x:.6f}" for x in qv) + "]"
                per_kind: dict[str, list[str]] = {}
                for kind in ("listing", "image", "property"):
                    cur.execute(
                        f"""SELECT listing_id FROM fl_chunks
                            WHERE kind = '{kind}'
                            ORDER BY embedding <=> '{vs}'::vector
                            LIMIT {PER_KIND_K}""")
                    per_kind[kind] = [r[0] for r in cur.fetchall()]
                fused = _rrf_fuse(per_kind)
                order = [lid for lid, _ in fused]
                r1 = order.index(want) + 1 if want in order else None
                hits1 += r1 == 1
                hits3 += r1 is not None and r1 <= 3
                mark = "OK " if r1 == 1 else ("~  " if r1 == 2 else "MISS")
                print(f"  [{mark}] rank={r1} want={want} q={q!r}")
                print(f"         fused order: {order[:5]}")
    print(f"\neval: hit@1 {hits1}/{len(EVAL)}  hit@3 {hits3}/{len(EVAL)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--step", default="all",
                    choices=["all", "download", "translate", "prep",
                             "embed", "load", "verify"])
    ap.add_argument("--dataset", default="bishkek",
                    choices=["bishkek", "flanders"],
                    help="bishkek: open RU corpus (default); "
                         "flanders: gated NL corpus")
    ap.add_argument("--image-set", default="full", choices=["full", "s224"],
                    help="full: gallery covers via 24.8GiB tars (recommended); "
                         "s224: light 224px shards (~1.7GB, _000 != cover)")
    ap.add_argument("--max-listings", type=int, default=0,
                    help="prep/embed only first N listings with photos (0=all)")
    ap.add_argument("--images-per-listing", type=int, default=0,
                    help="cap images per listing for quick tests (0=all images)")
    ap.add_argument("--dim", type=int, default=EG2_DIM,
                    help="MRL dim: 128/256/512/768 (default from EG2_DIM env or 256)")
    ap.add_argument("--backend", default="llama", choices=["llama", "st"],
                    help="llama: GPU via ./embed-server.sh (fast, default); "
                         "st: sentence-transformers on CPU (slow fallback)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--src-lang", default="nl",
                    help="translate step: source language code (nl, ru, …)")
    ap.add_argument("--trans-csv", default="",
                    help="translate step: CSV with descriptions "
                         "(default: Flanders descriptions.csv)")
    ap.add_argument("--trans-id-col", default="listing_id")
    ap.add_argument("--trans-text-col", default="description_clean")
    args = ap.parse_args()

    steps = ["download", "translate", "prep", "embed", "load", "verify"] \
        if args.step == "all" else [args.step]
    if "download" in steps:
        step_download(args.image_set, args.dataset, args.max_listings)
    if "translate" in steps:
        if args.dataset == "bishkek" and not args.trans_csv:
            args.trans_csv = os.path.join(RAW_BISHKEK, "listings.csv")
            args.trans_id_col = "listing_id"
            args.trans_text_col = "description"
            if args.src_lang == "nl":
                args.src_lang = "ru"
        csv_path = args.trans_csv or os.path.join(
            RAW, "tabular_data/descriptions.csv")
        step_translate(args.src_lang, csv_path,
                       args.trans_id_col, args.trans_text_col)
    if "prep" in steps:
        step_prep(args.image_set, args.max_listings, args.images_per_listing,
                  args.dataset)
    if "embed" in steps:
        step_embed(args.dim, args.batch, args.backend)
    if "load" in steps:
        step_load(args.dim)
    if "verify" in steps:
        step_verify(args.dim, args.backend)


if __name__ == "__main__":
    main()
