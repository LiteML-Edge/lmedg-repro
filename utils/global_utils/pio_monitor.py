#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pio_monitor.py — Opens the PlatformIO Serial Monitor robustly
- Reads monitor_port and monitor_speed from platformio.ini (from [env:NAME] or default_envs)
- Allows CLI overrides (--port, --baud)
- Waits for the port to appear after upload/reset (--wait N seconds)
- Forwards extra arguments to the monitor (for example, --extra --eol CRLF --echo)
- **Now records automatically to TXT** using the monitor `time` and `log2file` filters
- The optional `--no-log` flag disables this behavior
- `--help` shows usage examples
"""
from __future__ import annotations
import argparse, os, sys, subprocess, time, json
from pathlib import Path
import configparser

def _split(s: str | None) -> list[str]:
    if not s:
        return []
    parts = []
    for token in s.replace(",", " ").split():
        t = token.strip()
        if t:
            parts.append(t)
    return parts

def find_platformio_ini(project_dir: Path) -> Path | None:
    cand = project_dir / "platformio.ini"
    return cand if cand.exists() else None

def read_env_from_ini(ini_path: Path, env_name: str | None) -> tuple[str | None, dict]:
    cfg = configparser.ConfigParser()
    cfg.read(ini_path, encoding="utf-8")
    chosen = None
    if env_name:
        chosen = env_name
    else:
        if cfg.has_section("platformio") and cfg.has_option("platformio", "default_envs"):
            defaults = _split(cfg.get("platformio", "default_envs"))
            if defaults:
                chosen = defaults[0]
        if not chosen:
            envs = [s[len("env:"):] for s in cfg.sections() if s.startswith("env:")]
            if len(envs) == 1:
                chosen = envs[0]
    env_dict = {}
    if chosen and cfg.has_section(f"env:{chosen}"):
        sec = f"env:{chosen}"
        for key in ("monitor_port", "monitor_speed", "upload_port"):
            if cfg.has_option(sec, key):
                env_dict[key] = cfg.get(sec, key)
    return chosen, env_dict

def pio_device_list(project_dir: Path) -> list[dict]:
    cmd = [sys.executable, "-m", "platformio", "device", "list", "--json-output"]
    try:
        out = subprocess.check_output(cmd, cwd=project_dir)
        data = json.loads(out.decode("utf-8", errors="ignore") or "[]")
        return data if isinstance(data, list) else []
    except Exception:
        for exe in ("platformio", "pio"):
            try:
                out = subprocess.check_output([exe, "device", "list", "--json-output"], cwd=project_dir)
                data = json.loads(out.decode("utf-8", errors="ignore") or "[]")
                return data if isinstance(data, list) else []
            except Exception:
                continue
    return []

def wait_for_port(target_port: str, project_dir: Path, timeout: int) -> bool:
    if not target_port or timeout <= 0:
        return True
    end = time.time() + timeout
    print(f"[monitor] Waiting for port {target_port} to appear (timeout={timeout}s)...")
    while time.time() < end:
        devices = pio_device_list(project_dir)
        for d in devices:
            if str(d.get("port", "")).strip().lower() == target_port.strip().lower():
                print(f"[monitor] Port found: {target_port}")
                return True
        time.sleep(1.0)
    print(f"[monitor] WARNING: Port {target_port} not found after {timeout}s (continuing anyway).")
    return False

def _ensure_filter_args(extra: list[str], enable_log: bool) -> list[str]:
    if not extra:
        extra = []
    if any(arg == "--raw" for arg in extra):
        return extra
    if not enable_log:
        return extra
    existing_filters = set()
    i = 0
    while i < len(extra):
        if extra[i] in ("-f", "--filter", "--filters"):
            if i + 1 < len(extra):
                existing_filters.add(extra[i+1])
                i += 2
                continue
        i += 1
    def add_filter(name: str):
        if name not in existing_filters:
            extra.extend(["-f", name])
            existing_filters.add(name)
    add_filter("time")
    add_filter("log2file")
    return extra

def build_monitor_cmd(project_dir: Path, env_name: str | None, port: str | None, baud: int | None, extra: list[str] | None, enable_log: bool=True) -> list[str]:
    cmd = [sys.executable, "-m", "platformio", "device", "monitor", "-d", str(project_dir)]
    if env_name:
        cmd += ["-e", env_name]
    if port:
        cmd += ["--port", port]
    if baud:
        cmd += ["--baud", str(baud)]
    safe_extra = _ensure_filter_args(extra or [], enable_log)
    if safe_extra:
        cmd += safe_extra
    return cmd

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Open the PlatformIO Serial Monitor with automatic port/baud detection", add_help=False)
    ap.add_argument("-d","--project-dir", default="firmwares", help="PlatformIO project root directory containing platformio.ini")
    ap.add_argument("-e","--env", help="Environment name (for example, esp32dev). If omitted, use default_envs or the only environment in the INI file")
    ap.add_argument("--port", help="Serial port (for example, COM5 or /dev/ttyUSB0). If omitted, try to read it from the INI file")
    ap.add_argument("--baud", type=int, help="Baud rate (for example, 115200). If omitted, try to read it from the INI file (default: 115200)")
    ap.add_argument("--wait", type=int, default=15, help="Seconds to wait for the port to appear after upload/reset")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, help="Extra arguments forwarded to 'platformio device monitor'")
    ap.add_argument("--no-log", action="store_true", help="Do not add file-logging filters (time, log2file)")
    ap.add_argument("--help", action="store_true", help="Show usage examples")
    args = ap.parse_args(argv)

    if args.help:
        print("\nUsage examples:")
        print("  python pio_monitor.py                    # Default monitor with TXT logging")
        print("  python pio_monitor.py --no-log           # Monitor without file logging")
        print("  python pio_monitor.py --port COM5 --baud 9600")
        print("  python pio_monitor.py -e esp32dev        # Use the environment from platformio.ini")
        print("  python pio_monitor.py --wait 20          # Wait 20 seconds for the port to appear")
        print("  python pio_monitor.py --extra --eol CRLF --echo")
        print("  python pio_monitor.py -d firmwares -e esp32dev --port COM4 --baud 9600 --wait 10 --extra --raw --filter default")
        return 0

    project_dir = Path(args.project_dir).resolve()
    ini = find_platformio_ini(project_dir)
    env_name = args.env
    inferred_port = None
    inferred_baud = None
    if ini:
        env_name, env_cfg = read_env_from_ini(ini, env_name)
        inferred_port = env_cfg.get("monitor_port") if env_cfg else None
        inferred_baud = env_cfg.get("monitor_speed") if env_cfg else None
    port = args.port or inferred_port
    baud = args.baud or (int(inferred_baud) if inferred_baud else 115200)
    print(f"[monitor] project_dir={project_dir}")
    print(f"[monitor] env={env_name} port={port} baud={baud}")
    if args.extra:
        print(f"[monitor] extra={' '.join(args.extra)}")
    if args.no_log:
        print("[monitor] --no-log: logging filters disabled")
    if port:
        wait_for_port(port, project_dir, args.wait)
    cmd = build_monitor_cmd(project_dir, env_name, port, baud, args.extra, enable_log=not args.no_log)
    print("[monitor] Exec:", " ".join(cmd))
    try:
        return subprocess.call(cmd, cwd=project_dir)
    except FileNotFoundError:
        for exe in ("platformio", "pio"):
            try:
                cmd2 = [exe, "device", "monitor", "-d", str(project_dir)]
                if env_name: cmd2 += ["-e", env_name]
                if port:     cmd2 += ["--port", port]
                if baud:     cmd2 += ["--baud", str(baud)]
                if args.extra:
                    cmd2 += _ensure_filter_args(args.extra or [], enable_log=not args.no_log)
                else:
                    cmd2 += _ensure_filter_args([], enable_log=not args.no_log)
                print("[monitor] Exec(fallback):", " ".join(cmd2))
                return subprocess.call(cmd2, cwd=project_dir)
            except FileNotFoundError:
                continue
        print("[monitor] ERROR: PlatformIO CLI not found. Install it with: python -m pip install platformio", file=sys.stderr)
        return 127

if __name__ == "__main__":
    raise SystemExit(main())
