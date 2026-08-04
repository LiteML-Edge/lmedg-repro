\
#!/usr/bin/env python
\
from __future__ import annotations
import argparse, os, sys, subprocess
from pathlib import Path

def find_repo_root() -> Path:
    env_root = os.environ.get("RUNNER_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    here = Path(__file__).resolve()
    for base in [here, *here.parents, Path.cwd(), *Path.cwd().parents]:
        if (base / "utils").exists():
            return base
    return Path.cwd()

def run(cmd: list[str], cwd: Path | None = None) -> int:
    print(f"[pio] CMD: {' '.join(cmd)} (cwd={cwd or Path.cwd()})")
    try:
        r = subprocess.run(cmd, cwd=cwd, check=False)
        return r.returncode
    except FileNotFoundError as e:
        print("[pio] ERROR: command not found. Install the PlatformIO CLI (pip install platformio).", file=sys.stderr)
        return 127

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PlatformIO upload through the runner")
    ap.add_argument("--project-dir", default="firmwares")
    ap.add_argument("--selector", default=os.environ.get("PIO_MODEL_SELECTOR", "latest"))
    ap.add_argument("--port", help="serial port (for example, COM6)")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, help="extra arguments")
    args = ap.parse_args(argv)

    repo = find_repo_root()
    project_dir = (repo / args.project_dir).resolve()

    env = os.environ.copy()
    env["PIO_MODEL_SELECTOR"] = args.selector

    cmd = ["platformio", "run", "--target", "upload"]
    if args.port:
        cmd += ["--upload-port", args.port]
    if args.extra:
        cmd += args.extra
    print(f"[pio] Using selector: {args.selector}")
    try:
        r = subprocess.run(cmd, cwd=project_dir, env=env, check=False)
        return r.returncode
    except FileNotFoundError:
        cmd[0] = "pio"
        r = subprocess.run(cmd, cwd=project_dir, env=env, check=False)
        return r.returncode

if __name__ == "__main__":
    sys.exit(main())