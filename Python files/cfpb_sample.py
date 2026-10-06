"""
Stratified sample of CFPB consumer complaints WITH narratives.

CFPB stopped publishing narratives in the live database/API on Aug 14, 2026,
but released everything published up to then in its FOIA narratives archive:
https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/

This script downloads the archive files covering your 12-month window,
keeps complaints that have narrative text, and samples ~5,000 rows spread
evenly across products and months.

Usage:
    pip install requests pandas
    python cfpb_sample.py

Output: cfpb_sample.csv (same columns as the CFPB website export)
"""
import io
import os
import zipfile
import numpy as np
import pandas as pd
import requests

# ---- Settings you may want to change ------------------------------------
START = "2025-08-01"          # first day of the window
END = "2026-08-01"            # day after the last day of the window
TARGET_ROWS = 5000            # total rows in the final sample
POOL_PER_MONTH = 1000         # max random rows kept per product per month before final sampling
ALLOCATION = "equal"          # "equal" = same rows per product, "proportional" = mirrors real volume
SEED = 42
OUT_FILE = "cfpb_complaints_sample_2025-08_to_2026-07.csv"  # output CSV file
DOWNLOAD_DIR = "cfpb_archive"  # zip files are saved here so re-runs don't re-download

PRODUCTS = [
    "Credit reporting or other personal consumer reports",
    "Debt collection",
    "Credit card",
    "Checking or savings account",
    "Money transfer, virtual currency, or money service",
    "Mortgage",
    "Vehicle loan or lease",
    "Student loan",
    "Payday loan, title loan, personal loan, or advance loan",
    "Prepaid card",
    "Debt or credit management",
]
# --------------------------------------------------------------------------

BASE = "https://files.consumerfinance.gov/f/documents/"
# (first month, last month, file name) for each archive file
ARCHIVE_FILES = [
    ("2024-11", "2024-12", "CCDB_Export_8_November_2024_through_December_2024.zip"),
    ("2025-01", "2025-02", "CCDB_Export_9_January_2025_through_February_2025.zip"),
    ("2025-03", "2025-04", "CCDB_Export_10_March_2025_through_April_2025.zip"),
    ("2025-05", "2025-06", "CCDB_Export_11_May_2025_through_June_2025.zip"),
    ("2025-07", "2025-08", "CCDB_Export_12_July_2025_through_August_2025.zip"),
    ("2025-09", "2025-10", "CCDB_Export_13_September_2025_through_October_2025.zip"),
    ("2025-11", "2025-12", "CCDB_Export_14_November_2025_through_December_2025.zip"),
    ("2026-01", "2026-02", "CCDB_Export_15_January_2026_through_February_2026.zip"),
    ("2026-03", "2026-03", "CCDB_Export_16_March_2026.zip"),
    ("2026-04", "2026-04", "CCDB_Export_17_April_2026.zip"),
    ("2026-05", "2026-05", "CCDB_Export_18_May_2026.zip"),
    ("2026-06", "2026-06", "CCDB_Export_19_June_2026.zip"),
    ("2026-07", "2026-07", "CCDB_Export_20_July_2026.zip"),
    ("2026-08", "2026-08", "CCDB_Export_21_August_2026.zip"),
]

NARRATIVE = "Consumer complaint narrative"
DATE = "Date received"
PRODUCT = "Product"


def needed_files():
    first, last = START[:7], (pd.Timestamp(END) - pd.Timedelta(days=1)).strftime("%Y-%m")
    return [f for a, b, f in ARCHIVE_FILES if not (b < first or a > last)]


def download(name):
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    path = os.path.join(DOWNLOAD_DIR, name)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"  already downloaded: {name}")
        return path
    print(f"  downloading {name} ...")
    with requests.get(BASE + name, stream=True, timeout=600,
                      headers={"User-Agent": "research-sampling-script"}) as r:
        r.raise_for_status()
        done = 0
        with open(path + ".part", "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                done += len(chunk)
                print(f"\r    {done / 1e6:.0f} MB", end="")
    os.replace(path + ".part", path)
    print()
    return path


def read_archive(path, rng, pools):
    """Stream the CSV(s) inside the zip; keep a uniform random subset per product-month."""
    start, end = pd.Timestamp(START), pd.Timestamp(END)
    with zipfile.ZipFile(path) as z:
        for member in z.namelist():
            if not member.lower().endswith(".csv"):
                continue
            print(f"  reading {member}")
            with z.open(member) as fh:
                reader = pd.read_csv(io.TextIOWrapper(fh, encoding="utf-8", errors="replace"),
                                     dtype=str, chunksize=100_000)
                for chunk in reader:
                    if NARRATIVE not in chunk.columns:
                        print(f"    WARNING: no '{NARRATIVE}' column. Columns are: {chunk.columns.tolist()}")
                        return
                    text = chunk[NARRATIVE].fillna("").str.strip()
                    dates = pd.to_datetime(chunk[DATE], errors="coerce")
                    keep = (text != "") & (dates >= start) & (dates < end) & chunk[PRODUCT].isin(PRODUCTS)
                    chunk = chunk[keep].copy()
                    if chunk.empty:
                        continue
                    chunk["sample_month"] = dates[keep].dt.strftime("%Y-%m")
                    chunk["_rand"] = rng.random(len(chunk))
                    for key, g in chunk.groupby([PRODUCT, "sample_month"]):
                        pools[key] = pd.concat([pools.get(key), g]).nsmallest(POOL_PER_MONTH, "_rand")


def sample_by_month(pool, n):
    """Split n rows evenly across months; if a month runs short,
    its unused rows go to the months that still have data."""
    groups = {m: g for m, g in pool.groupby("sample_month")}
    quota = {m: 0 for m in groups}
    remaining, open_months = n, set(groups)
    while remaining > 0 and open_months:
        share = max(1, remaining // len(open_months))
        for m in sorted(open_months):
            take = min(share, len(groups[m]) - quota[m], remaining)
            quota[m] += take
            remaining -= take
            if quota[m] >= len(groups[m]):
                open_months.discard(m)
            if remaining == 0:
                break
    parts = [groups[m].sample(n=k, random_state=SEED) for m, k in quota.items() if k > 0]
    return pd.concat(parts, ignore_index=True)


def main():
    rng = np.random.default_rng(SEED)
    pools = {}

    files = needed_files()
    print(f"Window {START} to {END} needs {len(files)} archive files.")
    for name in files:
        print(f"\n{name}")
        read_archive(download(name), rng, pools)

    by_product = {}
    for (product, _month), g in pools.items():
        by_product.setdefault(product, []).append(g)
    by_product = {p: pd.concat(gs, ignore_index=True) for p, gs in by_product.items()}
    if not by_product:
        print("\nNo complaints with narratives found. Check the warnings above.")
        return

    # ---- Decide how many rows each product gets ----
    available = {p: len(df) for p, df in by_product.items()}
    if ALLOCATION == "proportional":
        total = sum(available.values())
        quota = {p: int(round(TARGET_ROWS * n / total)) for p, n in available.items()}
    else:
        quota = {p: 0 for p in available}
        remaining, open_products = TARGET_ROWS, set(available)
        while remaining > 0 and open_products:
            share = max(1, remaining // len(open_products))
            for p in sorted(open_products):
                take = min(share, available[p] - quota[p], remaining)
                quota[p] += take
                remaining -= take
                if quota[p] >= available[p]:
                    open_products.discard(p)
                if remaining == 0:
                    break

    # ---- Sample ----
    print()
    samples = []
    for p, n in quota.items():
        s = sample_by_month(by_product[p], n)
        samples.append(s)
        print(f"{p}: {len(s)} rows")
        print("   per month:", s["sample_month"].value_counts().sort_index().to_dict())

    final = pd.concat(samples, ignore_index=True).sample(frac=1, random_state=SEED)
    final = final.drop(columns=["_rand", "sample_month"])
    final.to_csv(OUT_FILE, index=False)

    mb = os.path.getsize(OUT_FILE) / 1e6
    print(f"\nSaved {len(final)} rows to {OUT_FILE} ({mb:.1f} MB)")


if __name__ == "__main__":
    main()