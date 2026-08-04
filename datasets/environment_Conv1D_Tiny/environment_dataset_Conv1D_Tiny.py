"""
Script: environment_dataset_Conv1D_Tiny.py
Module role:
    Generate the consolidated LiteML-Edge environmental dataset from the
    Singapore temperature and humidity source CSV files.

Technical summary:
    This script locates the Singapore temperature and humidity CSV files,
    detects the time column, separates indoor and outdoor sensor columns,
    computes row-wise indoor averages, preserves the single outdoor channels,
    merges both sources by timestamp, builds the final dataset, and exports
    the configured CSV output used by the LiteML-Edge workflow.

Execution model:
    The script is intentionally preserved as a top-level executable module.
    It performs the full dataset-generation workflow immediately when run.

Global execution support:
    The bootstrap resolves the project root through RUNNER_PROJECT_ROOT or by
    searching for the repository-level utils/ directory. This allows the script
    to run from any working directory inside the LiteML project environment,
    while preserving the original computational logic.

Inputs:
    - Singapore_Temp.csv
    - Singapore_RH.csv

Outputs:
    - environment_dataset_Conv1D_Tiny.csv
  
Notes:
    The computational logic, output naming, aggregation policy, and audit
    checks are preserved. The adjustments in this version are limited to
    project-root bootstrap robustness.
"""

# --- Bootstrap: ensure that `utils/` is importable locally and through the runner ---
from pathlib import Path
import pandas as pd
import sys, os
import numpy as np
import re

ROOT = os.environ.get("RUNNER_PROJECT_ROOT")
if not ROOT:
    HERE = Path(__file__).resolve()
    for base in [HERE, *HERE.parents, Path.cwd(), *Path.cwd().parents]:
        if (base / "utils").exists():
            ROOT = str(base)
            break
if ROOT and ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# -----------------------------------------------------------------------------------

# Try to use project paths; if unavailable, fall back to local path discovery.
try:
    from utils.global_utils.paths import PROJECT_ROOT, DATASET_ENVIRONMENT, DATASET_DATASET
except Exception:
    PROJECT_ROOT = None
    DATASET_ENVIRONMENT = None
    DATASET_DATASET = None

# =========================
# Configuration
# =========================
TIME_COL_CANDIDATES = ["Datetime", "datetime", "timestamp", "time", "date"]

def find_time_col(df: pd.DataFrame) -> str:
    """Return the first valid time column name among the accepted candidates."""
    for c in TIME_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"Time column not found. Available columns: {list(df.columns)}")

def split_indoor_outdoor(cols, prefix_pattern):
    """
    Split sensor columns into indoor and outdoor groups.

    Indoor columns end with a numeric suffix, such as _11.._15.
    Outdoor columns use the base name without a numeric suffix, such as T(C)
    or RH(%).
    """
    indoor = [c for c in cols if re.match(prefix_pattern, c) and re.search(r"_\d+$", c)]
    outdoor = [c for c in cols if re.match(prefix_pattern, c) and not re.search(r"_\d+$", c)]
    return indoor, outdoor

def safe_mean(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Compute a row-wise NaN-safe arithmetic mean over the selected columns."""
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    return X.mean(axis=1, skipna=True)

def _candidate_dirs():
    """Return candidate directories where the source CSV files may be located."""
    here = Path(__file__).resolve().parent
    cand = [here, here / "dataset_generator", here.parent, here.parent / "dataset_generator"]
    if DATASET_DATASET is not None:
        cand += [Path(DATASET_DATASET), Path(DATASET_DATASET) / "dataset_generator"]
    if DATASET_ENVIRONMENT is not None:
        cand += [Path(DATASET_ENVIRONMENT), Path(DATASET_ENVIRONMENT) / "dataset_generator"]
    # Remove duplicates while preserving order.
    seen = set()
    out = []
    for d in cand:
        d = d.resolve()
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out

def find_file(relative_path: str) -> Path:
    """Locate an input file using project-aware candidate directories."""
    rel = Path(relative_path)

    for d in _candidate_dirs():
        p = (d / rel).resolve()
        if p.exists():
            return p

    # Fallback: limited recursive search near the script location.
    here = Path(__file__).resolve().parent
    target_name = rel.name

    for hit in here.rglob(target_name):
        if hit.name == rel.name:
            return hit.resolve()

    raise FileNotFoundError(
        f"Could not find {relative_path}. Searched in:\n" +
        "\n".join([f" - {d}" for d in _candidate_dirs()])
    )

def _sensor_suffix_key(col: str):
    """
    Sort indoor sensor columns by their numeric suffix, such as _11, _12, and _15.

    Columns without a numeric suffix are placed last, although they are not
    expected in the indoor subset.
    """
    m = re.search(r"_(\d+)$", col)
    return int(m.group(1)) if m else 10**9

# Inputs
TEMP_CSV = find_file("singapore_dataset/Singapore_Temp.csv")
RH_CSV   = find_file("singapore_dataset/Singapore_RH.csv")

# =========================
# Output (same folder as the script by default)
# =========================
OUT_DIR = Path(__file__).resolve().parent
OUT_CSVS = [
    (OUT_DIR / "environment_dataset_Conv1D_Tiny.csv").resolve(),
]

# =========================
# Load
# =========================
dfT = pd.read_csv(TEMP_CSV)
dfH = pd.read_csv(RH_CSV)

tcol_T = find_time_col(dfT)
tcol_H = find_time_col(dfH)

# Deterministic parsing; preserves the current behavior without a fixed format.
dfT[tcol_T] = pd.to_datetime(dfT[tcol_T], errors="coerce")
dfH[tcol_H] = pd.to_datetime(dfH[tcol_H], errors="coerce")

nT_before_time = len(dfT)
nH_before_time = len(dfH)

dfT = dfT.dropna(subset=[tcol_T]).copy()
dfH = dfH.dropna(subset=[tcol_H]).copy()

nT_after_time = len(dfT)
nH_after_time = len(dfH)

# Normalize the time-column name to "datetime".
dfT = dfT.rename(columns={tcol_T: "datetime"})
dfH = dfH.rename(columns={tcol_H: "datetime"})

# =========================
# Detect indoor/outdoor columns
# =========================
temp_indoor_cols, temp_outdoor_cols = split_indoor_outdoor(dfT.columns, r"^T\(C\)")
rh_indoor_cols,   rh_outdoor_cols   = split_indoor_outdoor(dfH.columns, r"^RH\(%\)")

if len(temp_outdoor_cols) != 1 or len(rh_outdoor_cols) != 1:
    raise ValueError(
        "Expected exactly one outdoor column without suffix for temperature and humidity.\n"
        f"T_outdoor={temp_outdoor_cols}\nH_outdoor={rh_outdoor_cols}"
    )

if len(temp_indoor_cols) == 0 or len(rh_indoor_cols) == 0:
    raise ValueError(
        "Indoor columns with _NN suffix were not found.\n"
        f"T_indoor={temp_indoor_cols}\nH_indoor={rh_indoor_cols}"
    )

# Deterministic and auditable indoor-column ordering.
temp_indoor_cols = sorted(temp_indoor_cols, key=_sensor_suffix_key)
rh_indoor_cols   = sorted(rh_indoor_cols,   key=_sensor_suffix_key)

T_OUT_COL = temp_outdoor_cols[0]  # "T(C)"
H_OUT_COL = rh_outdoor_cols[0]    # "RH(%)"

# =========================
# Merge by datetime
# =========================
nT = len(dfT)
nH = len(dfH)

df = pd.merge(dfT, dfH, on="datetime", how="inner")
df = df.sort_values("datetime").reset_index(drop=True)

n_merge = len(df)

# =========================
# Aggregate indoor
# =========================
T_in = safe_mean(df, temp_indoor_cols).astype(np.float32)
H_in = safe_mean(df, rh_indoor_cols).astype(np.float32)

# Outdoor (single column)
T_out = pd.to_numeric(df[T_OUT_COL], errors="coerce").astype(np.float32)
H_out = pd.to_numeric(df[H_OUT_COL], errors="coerce").astype(np.float32)

# =========================
# Build final dataset (required columns only, fixed order)
# =========================
out = pd.DataFrame({
    "datetime": df["datetime"],
    "T_out": T_out,
    "T_in":  T_in,
    "H_out": H_out,
    "H_in":  H_in,
})

n_out_before_drop = len(out)

# Remove invalid rows when any required field is missing.
out = out.dropna(subset=["datetime", "T_out", "T_in", "H_out", "H_in"]).copy()

n_out_after_drop = len(out)

# =========================
# Save configured outputs + equivalence proof when multiple outputs are enabled
# =========================
for p in OUT_CSVS:
    out.to_csv(p, index=False, encoding="utf-8-sig", float_format="%.2f")

def _sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    """Return the SHA256 digest of a file."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def _files_equal(a: Path, b: Path, chunk: int = 1024 * 1024) -> bool:
    """Return True when two files are byte-identical."""
    if a.stat().st_size != b.stat().st_size:
        return False
    with open(a, "rb") as fa, open(b, "rb") as fb:
        while True:
            ba = fa.read(chunk)
            bb = fb.read(chunk)
            if ba != bb:
                return False
            if not ba:
                return True

hashes = {str(p): _sha256(p) for p in OUT_CSVS}

pairwise_equivalence = []
if len(OUT_CSVS) > 1:
    base_output = OUT_CSVS[0]
    for idx, candidate in enumerate(OUT_CSVS[1:], start=1):
        pairwise_equivalence.append((f"base == copy{idx}", _files_equal(base_output, candidate)))
    all_equal = (
        all(result for _, result in pairwise_equivalence) and
        len({hashes[str(p)] for p in OUT_CSVS}) == 1
    )
else:
    all_equal = True

print("OK ✅")
print("TEMP_CSV:", TEMP_CSV)
print("RH_CSV:  ", RH_CSV)

print("\n=== Row audit ===")
print(f"Temp:  read={nT_before_time} | valid datetime={nT_after_time}")
print(f"RH:    read={nH_before_time} | valid datetime={nH_after_time}")
print(f"After datetime drop: Temp={nT} | RH={nH}")
print(f"Inner merge (intersection): {n_merge}")
print(f"Before final drop (NaN in variables): {n_out_before_drop}")
print(f"After final drop: {n_out_after_drop}")
print(f"Removed in final drop: {n_out_before_drop - n_out_after_drop}")

print("\n=== Detected columns ===")
print("Indoor T cols:", temp_indoor_cols)
print("Outdoor T col:", T_OUT_COL)
print("Indoor H cols:", rh_indoor_cols)
print("Outdoor H col:", H_OUT_COL)

print("\nOutputs:")
for p in OUT_CSVS:
    print(" -", p)
print("Rows:", len(out))

print("\n=== Byte-level equivalence proof ===")
if pairwise_equivalence:
    for label, result in pairwise_equivalence:
        print(f"{label}:", result)
else:
    print("Single output configured; pairwise equivalence check skipped.")

print("\n=== SHA256 ===")
for k, v in hashes.items():
    print(k, "->", v)

print("\nFINAL EQUIVALENCE:", "PASS" if all_equal else "FAIL")
