# utils/versioning.py
# How to use it:
#from pathlib import Path
#from utils.versioning import list_runs, resolve_run, resolve_latest

#BASE = Path("artifacts/base_model")

# List runs, newest first
#print(list_runs(BASE))

# Select the latest run
#run = resolve_latest(BASE)
# alternatively: run = resolve_run(BASE, "latest")

# Select a specific run
#run = resolve_run(BASE, "v003")             # by counter
#run = resolve_run(BASE, "ts20250819-143512")# by timestamp
#run = resolve_run(BASE, "-2")               # second newest
#run = resolve_run(BASE, r"C:\path\to\run.v007")  # direct path

from __future__ import annotations
from pathlib import Path
from typing import Union
from datetime import datetime
import os, json, re

PathLike = Union[str, os.PathLike]

def _is_file(p: Path) -> bool:
    return ''.join(p.suffixes) != ''

def next_version_path(base: PathLike, tag="v", width=3) -> Path:
    base = Path(base)
    parent = base.parent if str(base.parent) != "" else Path(".")
    is_file = _is_file(base)
    stem, ext = (base.stem, ''.join(base.suffixes)) if is_file else (base.name, "")
    if not base.exists():
        return base
    pat = re.compile(rf"^{re.escape(stem)}\.{re.escape(tag)}(\d{{{width}}}){re.escape(ext)}$")
    max_n = 0
    for child in parent.iterdir():
        m = pat.match(child.name)
        if m:
            try:
                max_n = max(max_n, int(m.group(1)))
            except ValueError:
                pass
    n = max_n + 1
    name = f"{stem}.{tag}{n:0{width}d}{ext}"
    return parent / name

def timestamp_path(base: PathLike, tag="ts", fmt="%Y%m%d-%H%M%S") -> Path:
    base = Path(base)
    is_file = _is_file(base)
    parent = base.parent if str(base.parent) != "" else Path(".")
    stem, ext = (base.stem, ''.join(base.suffixes)) if is_file else (base.name, "")
    ts = datetime.now().strftime(fmt)
    cand = parent / (f"{stem}.{tag}{ts}{ext}" if is_file else f"{stem}.{tag}{ts}")
    if not cand.exists():
        return cand
    i = 1
    while True:
        c = cand.with_name(cand.stem + f"-{i:03d}" + (ext if is_file else ""))
        if not c.exists():
            return c
        i += 1

def ensure_dir(path: PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def create_versioned_dir(base_dir: PathLike, strategy="timestamp") -> Path:
    base_dir = ensure_dir(base_dir)
    if strategy == "timestamp":
        run = timestamp_path(Path(base_dir) / "run")
    else:
        run = next_version_path(Path(base_dir) / "run")
    run.mkdir(parents=True, exist_ok=False)
    return run

def update_latest(target: PathLike, name="latest") -> Path:
    target = Path(target).resolve()
    link = target.parent / name
    try:
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target.name, target_is_directory=target.is_dir())
        return link
    except OSError:
        txt = link.with_suffix(".txt")
        txt.write_text(str(target), encoding="utf-8")
        return txt

def write_manifest(out_dir: Path, **info) -> Path:
    out_dir = ensure_dir(out_dir)
    manifest = Path(out_dir) / "manifest.json"
    manifest.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest

# ---------------------- Run resolution ----------------------
def list_runs(base_dir: Path) -> list[Path]:
    runs = [d for d in Path(base_dir).glob("run.*") if d.is_dir()]
    runs.sort(key=lambda d: d.stat().st_mtime, reverse=True)  # newest first
    return runs

def _resolve_latest_txt(base_dir: Path) -> Path | None:
    txt = (Path(base_dir) / "latest").with_suffix(".txt")
    if not txt.exists():
        return None
    raw = txt.read_text(encoding="utf-8").strip()
    p = Path(raw)
    return p if p.is_absolute() else (Path(base_dir) / p).resolve()

def resolve_latest(base_dir: Path) -> Path:
    base_dir = Path(base_dir)
    link = base_dir / "latest"
    if link.is_symlink():
        return link.resolve()
    p = _resolve_latest_txt(base_dir)
    if p and p.exists():
        return p
    runs = list_runs(base_dir)
    if not runs:
        raise FileNotFoundError(f"No runs found under {base_dir}")
    return runs[0]

def resolve_run(base_dir: Path, selector: str | None = "latest") -> Path:
    base_dir = Path(base_dir)

    if selector in (None, "", "latest"):
        return resolve_latest(base_dir)

    sel_path = Path(selector)
    if sel_path.exists():  # already a valid path
        return sel_path.resolve()

    cand = base_dir / selector
    if cand.exists():
        return cand.resolve()

    if re.fullmatch(r"v\d{3}", selector):            # example: v003
        cand = base_dir / f"run.{selector}"
        if cand.exists(): return cand.resolve()

    if re.fullmatch(r"ts\d{8}-\d{6}", selector):     # example: ts20250819-143512
        cand = base_dir / f"run.{selector}"
        if cand.exists(): return cand.resolve()

    if re.fullmatch(r"-\d+", selector):              # example: -1 (second newest), -2 (third newest)
        idx = int(selector[1:])
        runs = list_runs(base_dir)
        if 1 <= idx <= len(runs):
            return runs[idx].resolve()

    raise FileNotFoundError(f"Run '{selector}' not found under {base_dir}")
