# runner.py

from __future__ import annotations
import argparse
from argparse import RawTextHelpFormatter
import os
import sys
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
from pathlib import Path

# === CLI help and epilogue ===
HELP_EPILOG = r"""Stage selection (use names exactly as declared in the pipeline YAML):

  --only STEP              Run only STEP. Upstream dependencies are included by default.
  --no-upstream            Used with --only: do not run dependencies; run STEP in isolation.

  --from STEP              Run from STEP to the end, including required dependencies.
  --to STEP                Run from the beginning through STEP, including required dependencies.
  --after STEP             Run only stages that depend directly or indirectly on STEP.
  --before STEP            Run only stages required directly or indirectly by STEP.

  --keep-going             Continue after a failed stage and report failures at the end.
  --list                   Print the calculated execution plan and exit.
  --dry-run                Print the plan and complete commands without executing them.

Precedence rules when options are combined:
  1) --only, with or without --no-upstream, takes precedence over all other filters.
  2) Range filters are applied next: --from, --to, --after, and --before.
  3) --list prints the final plan and exits without execution.
  4) --dry-run prints the final plan and commands without execution.

Dependency behavior:
  - Without --no-upstream, required dependencies are always included.
  - With --no-upstream, the selected stage runs alone and prerequisites are not guaranteed.

Firmware-mode requirement:
  - runner.py does not change LITEML_MODE in include/config.h.
  - Select LITEML_MODE_REPLAY before reproducing the formal 1:1 validation builds.
  - Select LITEML_MODE_FIELD only for live-sensor execution and deployment-cost acquisition.

Complete examples:
  # 1) Inspect a pipeline without executing it
  python runner.py -p environment_esp32_mlp_pipeline.yaml --list
  python runner.py -p environment_esp32_mlp_pipeline.yaml --dry-run

  # 2) Build the six retained REPLAY configurations from the packaged firmware inputs
  python runner.py -p environment_esp32_mlp_pipeline.yaml --only pio_build --no-upstream
  python runner.py -p environment_stm32_mlp_pipeline.yaml --only pio_build --no-upstream
  python runner.py -p environment_esp32_Conv1D_pipeline.yaml --only pio_build --no-upstream
  python runner.py -p environment_stm32_Conv1D_pipeline.yaml --only pio_build --no-upstream
  python runner.py -p environment_esp32_lstm_pipeline.yaml --only pio_build --no-upstream
  python runner.py -p environment_stm32_lstm_pipeline.yaml --only pio_build --no-upstream

  # 3) Build, upload, and monitor one ESP32 configuration in separate stages
  python runner.py -p environment_esp32_mlp_pipeline.yaml --only pio_build --no-upstream
  python runner.py -p environment_esp32_mlp_pipeline.yaml --only pio_upload --no-upstream
  python runner.py -p environment_esp32_mlp_pipeline.yaml --only pio_monitor --no-upstream

  # 4) Build, upload, and monitor one STM32 configuration in separate stages
  python runner.py -p environment_stm32_mlp_pipeline.yaml --only pio_build --no-upstream
  python runner.py -p environment_stm32_mlp_pipeline.yaml --only pio_upload --no-upstream
  python runner.py -p environment_stm32_mlp_pipeline.yaml --only pio_monitor --no-upstream

  # 5) Run build and upload as a bounded range; monitor remains a separate interactive stage
  python runner.py -p environment_esp32_mlp_pipeline.yaml --from pio_build --to pio_upload
  python runner.py -p environment_esp32_mlp_pipeline.yaml --only pio_monitor --no-upstream

  # 6) Run a complete model-to-device pipeline
  python runner.py -p environment_esp32_mlp_pipeline.yaml

  # 7) Run only pio_build with all required upstream stages
  python runner.py -p environment_esp32_mlp_pipeline.yaml --only pio_build

  # 8) Run from training through the end and continue collecting independent failures
  python runner.py -p environment_esp32_mlp_pipeline.yaml --from base_training --keep-going

Operational notes:
  - Isolated pio_build requires the synchronized model, scaler, and Rolling-24 headers.
  - Isolated pio_upload requires a successful build and a connected target.
  - Isolated pio_monitor requires the correct serial port and firmware already running.
  - pio_monitor is interactive and remains active until stopped or the monitor process exits.

Retained build evidence:
  The six build records correspond to the REPLAY firmware configurations used for stage-wise
  conformance verification. FIELD configurations use the same PlatformIO projects and build
  procedure, with the acquisition mode selected by a compile-time macro. See
  environment_reports/final_builds/README.md.
"""


try:
    import yaml
except Exception:
    print("[runner] ERROR: PyYAML was not found. Install it with:\n"
          "  python -m pip install pyyaml", file=sys.stderr)
    sys.exit(1)


# ============================
# Data model
# ============================

@dataclass
class Step:
    name: str
    script: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    cwd: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)
    continue_on_error: bool = False

    def cmd(self, python_exe: str) -> List[str]:
        # Always use the current Python executable to avoid launcher mismatches.
        return [python_exe, str(self.script), *self.args]


@dataclass
class Pipeline:
    steps: Dict[str, Step]
    order: List[str]  # Declarative order for dependency-free cases.

    @staticmethod
    def load(yaml_path: Path) -> "Pipeline":
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        raw_steps = data.get("steps") or data.get("pipeline") or data
        if not isinstance(raw_steps, list):
            raise ValueError("Invalid YAML file: expected 'steps: [...]'.")

        steps: Dict[str, Step] = {}
        order: List[str] = []
        for item in raw_steps:
            name = item["name"]
            order.append(name)
            steps[name] = Step(
                name=name,
                script=item["script"],
                args=list(item.get("args", [])),
                env=dict(item.get("env", {})),
                cwd=item.get("cwd"),
                depends_on=list(item.get("depends_on", [])),
                continue_on_error=bool(item.get("continue_on_error", False)),
            )
        return Pipeline(steps=steps, order=order)

    def graph(self) -> Dict[str, Set[str]]:
        """Return dependency-to-stage adjacency."""
        adj: Dict[str, Set[str]] = {name: set() for name in self.steps}
        for s in self.steps.values():
            for dep in s.depends_on:
                if dep not in self.steps:
                    raise KeyError(f"Stage '{s.name}' depends on missing stage '{dep}'.")
                adj.setdefault(dep, set()).add(s.name)
        return adj

    def reverse_graph(self) -> Dict[str, Set[str]]:
        """Return stage-to-dependency adjacency for upstream traversal."""
        rev: Dict[str, Set[str]] = {name: set() for name in self.steps}
        for s in self.steps.values():
            rev[s.name] = set(s.depends_on)
        return rev


# ============================
# Selection and ordering algorithms
# ============================

def topo_sort(subset: Set[str], rev_graph: Dict[str, Set[str]]) -> List[str]:
    """Topologically sort a subset while preserving contained dependencies."""
    indeg = {n: 0 for n in subset}
    children = {n: set() for n in subset}
    for n in subset:
        for d in rev_graph[n]:
            if d in subset:
                indeg[n] += 1
                children.setdefault(d, set()).add(n)

    ready = [n for n in subset if indeg[n] == 0]
    plan: List[str] = []
    while ready:
        n = sorted(ready)[0]
        ready.remove(n)
        plan.append(n)
        for ch in children.get(n, set()):
            indeg[ch] -= 1
            if indeg[ch] == 0:
                ready.append(ch)

    if len(plan) != len(subset):
        missing = subset - set(plan)
        raise RuntimeError(f"Cycle detected or dependencies outside the subset: {missing}")
    return plan


def collect_downstream(start: Set[str], graph: Dict[str, Set[str]]) -> Set[str]:
    """Traverse dependency-to-stage edges and collect all downstream stages."""
    out = set()
    frontier = list(start)
    while frontier:
        n = frontier.pop(0)
        if n in out:
            continue
        out.add(n)
        for ch in graph.get(n, set()):
            frontier.append(ch)
    return out


def collect_upstream(targets: Set[str], rev_graph: Dict[str, Set[str]]) -> Set[str]:
    """Collect all direct and indirect upstream dependencies."""
    out = set()
    frontier = list(targets)
    while frontier:
        n = frontier.pop(0)
        if n in out:
            continue
        out.add(n)
        for d in rev_graph.get(n, set()):
            frontier.append(d)
    return out


def select_plan(pipeline: Pipeline, args: argparse.Namespace) -> List[str]:
    names = set(pipeline.steps.keys())
    g  = pipeline.graph()
    rg = pipeline.reverse_graph()

    # Default: run the entire DAG in topological order.
    target_set: Set[str] = set(names)

    # Main filters
    if args.only:
        only_set = set(args.only)
        unknown = only_set - names
        if unknown:
            raise KeyError(f"--only contains unknown stage(s): {sorted(unknown)}")
        if args.no_upstream:
            # Run only the explicitly selected stages.
            target_set = only_set
        else:
            # Include required upstream dependencies.
            target_set = collect_upstream(only_set, rg)
    elif args.from_step or args.to_step or args.after or args.before:
        target_set = set()
        if args.from_step:
            if args.from_step not in names:
                raise KeyError(f"--from references unknown stage '{args.from_step}'")
            target_set |= collect_downstream({args.from_step}, g)
        if args.to_step:
            if args.to_step not in names:
                raise KeyError(f"--to references unknown stage '{args.to_step}'")
            # Include every dependency required to reach the --to stage.
            target_set |= collect_upstream({args.to_step}, rg)
        if args.after:
            if args.after not in names:
                raise KeyError(f"--after references unknown stage '{args.after}'")
            downstream = collect_downstream({args.after}, g)
            target_set |= (downstream - {args.after})
        if args.before:
            if args.before not in names:
                raise KeyError(f"--before references unknown stage '{args.before}'")
            # Include every upstream dependency of the --before stage.
            target_set |= (collect_upstream({args.before}, rg) - {args.before})

        if not target_set:
            # No effective filter produced a plan.
            raise ValueError("The supplied filters did not select any stage.")
    else:
        # No filters: run all stages.
        target_set = names

    # Topologically sort the selected subset.
    plan = topo_sort(target_set, rg)

    # When --only and --no-upstream are combined, preserve the user-supplied order,
    # after validating the selection with the topological sorter.
    if args.only and args.no_upstream:
        # Order the plan exactly as provided in --only.
        order_map = {name: i for i, name in enumerate(args.only)}
        plan = sorted([n for n in plan if n in order_map], key=lambda n: order_map[n])

    # When --from and --to are combined, restrict execution to that window.
    if args.from_step and args.to_step:
        i_from = plan.index(args.from_step) if args.from_step in plan else 0
        i_to   = plan.index(args.to_step)   if args.to_step in plan else len(plan)-1
        if i_from > i_to:
            raise ValueError(f"--from '{args.from_step}' appears after --to '{args.to_step}' in the plan.")
        plan = plan[i_from:i_to+1]

    # Apply --after and --before as additional plan boundaries.
    if args.after and args.after in plan:
        idx = plan.index(args.after)
        plan = plan[idx+1:]
    if args.before and args.before in plan:
        idx = plan.index(args.before)
        plan = plan[:idx]

    return plan


# ============================
# Execution
# ============================

PROJECT_ROOT = Path(__file__).resolve().parent

def fmt_ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def run_step(step: Step, project_root: Path, keep_going: bool, dry_run: bool) -> int:
    # Prepare the environment by inheriting the current process and applying stage overrides.
    env = os.environ.copy()
    env.update({k: str(v) for k, v in step.env.items()})
    # Always expose the project root to stage scripts.
    env.setdefault("RUNNER_PROJECT_ROOT", str(project_root))

    # Working directory
    if step.cwd in (None, "", "null"):
        cwd = project_root
    else:
        # Resolve relative paths against the project root.
        cwd = Path(step.cwd)
        if not cwd.is_absolute():
            cwd = project_root / cwd
        cwd = cwd.resolve()

    # Stage script path
    script_path = Path(step.script)
    if not script_path.is_absolute():
        script_path = (project_root / script_path).resolve()

    if not script_path.exists():
        print(f"[{fmt_ts()}] [ERROR] Stage script not found: {script_path}")
        return 1

    cmd = step.cmd(sys.executable)
    print(f"[{fmt_ts()}] [RUN] Running '{step.name}': {cmd} (cwd={cwd})")
    if dry_run:
        return 0

    try:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env)
        rc = proc.returncode
    except KeyboardInterrupt:
        print(f"[{fmt_ts()}] [STOP] Interrupted by the user during '{step.name}'.")
        return 130

    status = "OK" if rc == 0 else f"ERROR({rc})"
    print(f"[{fmt_ts()}] [DONE] Finished '{step.name}' -> {status}")
    if rc != 0 and not (keep_going or step.continue_on_error):
        print("[runner] Stopping after failure. Use --keep-going to continue.")
    return rc


def print_plan(pipeline: Pipeline, plan: List[str]):
    print("Execution plan:")
    for i, name in enumerate(plan, 1):
        s = pipeline.steps[name]
        deps = s.depends_on or []
        print(f"  {i:02d}. {name} -> {s.script} deps={deps}")


def main():
    ap = argparse.ArgumentParser(description="Pipeline runner (DAG)", epilog=HELP_EPILOG, formatter_class=RawTextHelpFormatter)
    ap.add_argument("--pipeline", "-p", default="pipeline.yaml", help="Pipeline YAML file")
    sel = ap.add_argument_group("Selection")
    sel.add_argument("--only", nargs="+", help="Run only the selected stage(s). Dependencies are included by default.")
    sel.add_argument("--no-upstream", action="store_true", help="Used with --only: do not include dependencies; run exactly the selected stages.")
    sel.add_argument("--from", dest="from_step", help="Initial stage, inclusive, and everything downstream")
    sel.add_argument("--to", dest="to_step", help="Run through this stage, inclusive")
    sel.add_argument("--after", help="Run only stages after X, excluding X")
    sel.add_argument("--before", help="Run only stages before Y, excluding Y")
    ap.add_argument("--list", action="store_true", help="Print the plan without executing it")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    ap.add_argument("--keep-going", action="store_true", help="Continue even if a stage fails")
    args = ap.parse_args()

    yml_path = Path(args.pipeline)
    if not yml_path.is_file():
        print(f"[runner] ERROR: pipeline not found: {yml_path}")
        return 2

    pipeline = Pipeline.load(yml_path)
    plan = select_plan(pipeline, args)

    print_plan(pipeline, plan)
    if args.list:
        return 0

    if args.dry_run:
        overall_rc = 0
        for name in plan:
            step = pipeline.steps[name]
            rc = run_step(
                step,
                project_root=PROJECT_ROOT,
                keep_going=args.keep_going,
                dry_run=True,
            )
            if rc != 0:
                overall_rc = rc
                if not (args.keep_going or step.continue_on_error):
                    break
        return overall_rc

    # Execute
    overall_rc = 0
    for name in plan:
        step = pipeline.steps[name]
        rc = run_step(
            step,
            project_root=PROJECT_ROOT,
            keep_going=args.keep_going,
            dry_run=False,
        )
        if rc != 0:
            overall_rc = rc
            if not (args.keep_going or step.continue_on_error):
                break
    return overall_rc


if __name__ == "__main__":
    sys.exit(main())
