#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# make_dirs_from_plan.py — creates only the DIRECTORY TREE from a .txt plan
#
# Simple plan syntax (plan.txt):
#   - One folder per line, relative to --root
#   - Comments start with '#'
#   - Blank lines are ignored
#   - A trailing '/' is optional
#   - Also accepts plans in 'tree' format (with '├──', '└──', etc.); markers are ignored
#
# plan.txt example:
#     project/
#       data/
#       src/
#         models/
#       notebooks/
#     logs/
#     artifacts/
#
# Usage:
#     python make_dirs_from_plan.py --root ./workspace --plan plan.txt
#     python make_dirs_from_plan.py --root . --plan plan.txt --dry-run --verbose
from __future__ import annotations
import argparse, sys, re
from pathlib import Path

TREE_PREFIX_RE = re.compile(r'''
    ^[ \t\|\u2502]*               # spaces / tabs / vertical bars │
    (?:[-+*] | \u251c\u2500\u2500 | \u2514\u2500\u2500)?  # optional -, +, *, ├──, └── markers
    \s*
''', re.VERBOSE)

def clean_line(line: str) -> str:
    """Remove comments, 'tree' markers, and spaces; return a relative path (or '')."""
    s = line.strip()
    if not s or s.startswith('#'):
        return ''
    # Remove tree-drawing markers (├──, └──, │, -, etc.)
    s = TREE_PREFIX_RE.sub('', s)
    # Remove optional quotes and normalize separators
    s = s.strip().strip('\"\'').strip()
    # Remove a trailing ':' (sometimes present in tree exports)
    if s.endswith(':'):
        s = s[:-1].strip()
    # Remove a trailing '/'
    s = s.rstrip('/\\').strip()
    return s

def ensure_within_root(root: Path, target: Path) -> Path:
    """Ensure that 'target' remains inside 'root'."""
    root = root.resolve()
    target = target.resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise ValueError(f"Input points outside the root: {target} (root={root})")
    return target

def create_dirs(root: Path, plan_file: Path, dry_run: bool=False, verbose: bool=False) -> list[Path]:
    created = []
    with plan_file.open('r', encoding='utf-8') as f:
        for ln, raw in enumerate(f, 1):
            rel = clean_line(raw)
            if not rel:
                continue
            # Ignore lines that look like files (a dot follows the final separator)
            # Remove this block to allow dots in folder names.
            name = Path(rel).name
            if '.' in name and not name.startswith('.'):
                if verbose:
                    print(f"[skip:line {ln}] looks like a file: {rel}")
                continue
            dest = (root / rel)
            dest = ensure_within_root(root, dest)
            if verbose or dry_run:
                print(f"[dir] {dest}")
            if not dry_run:
                dest.mkdir(parents=True, exist_ok=True)
                created.append(dest)
    return created

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Create a DIRECTORY TREE from a simple .txt plan')
    ap.add_argument('--root', required=True, help='Root directory where the tree will be created')
    ap.add_argument('--plan', required=True, help='Text file containing the directory list/tree')
    ap.add_argument('--dry-run', action='store_true', help='Create nothing; only show what would be created')
    ap.add_argument('--verbose', action='store_true', help='Print each processed directory')
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    plan = Path(args.plan).resolve()

    if not plan.exists():
        print(f'[error] Plan not found: {plan}', file=sys.stderr)
        return 2

    root.mkdir(parents=True, exist_ok=True)
    created = create_dirs(root, plan, dry_run=args.dry_run, verbose=args.verbose)

    print(f"\nOK - {'simulation completed' if args.dry_run else f'{len(created)} directory/directories created or confirmed'} at: {root}")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
