"""CLI facade for benchmark suite runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from tools.monitor.progress_monitor import LiveProgressMonitor
from tools.suite_runner.process import (
    build_run_overrides,
    build_suite_context,
    run_wave,
    start_suite,
    update_live_suite_manifest,
)
from tools.suite_runner.reports import write_suite_outputs
from tools.suite_runner.spec import (
    group_runs_by_repeat,
    load_suite,
    resolve_suite_path,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark suite.")
    parser.add_argument("--suite", required=True, help="Suite YAML path.")
    parser.add_argument("--results-dir", default="results", help="Results root.")
    parser.add_argument("--port", type=int, default=8101, help="Base game server port.")
    parser.add_argument("--max-parallel", default=5, type=int, help="Max concurrent runs.")
    parser.add_argument(
        "--real-time",
        action="store_true",
        help="Compatibility alias for --clock realtime.",
    )
    parser.add_argument(
        "--clock",
        choices=("paused", "realtime", "final-rt"),
        default=None,
        help="Interaction clock override for every run.",
    )
    parser.add_argument(
        "--final-rt",
        action="store_true",
        help="Alias for --clock final-rt.",
    )
    parser.add_argument(
        "--exclude-game",
        action="append",
        default=[],
        help="Exclude a game id after suite expansion; repeatable.",
    )
    parser.add_argument(
        "--only-game",
        action="append",
        default=[],
        help="Keep only these game ids after suite expansion; repeatable.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = Path(__file__).resolve().parent
    suite_path = resolve_suite_path(args.suite)
    results_dir = Path(args.results_dir)

    suite = load_suite(suite_path)
    selected_runs = [
        dict(run)
        for run in suite.runs
        if (not args.only_game or run["game_id"] in set(args.only_game))
        and run["game_id"] not in set(args.exclude_game)
    ]
    if len(selected_runs) != len(suite.runs):
        if not selected_runs:
            raise ValueError("Suite filters removed every run")
        for index, run in enumerate(selected_runs, start=1):
            run["run_index"] = index
        suite = replace(
            suite,
            runs=selected_runs,
            repeat_waves=group_runs_by_repeat(selected_runs),
        )
    if args.final_rt:
        if args.clock not in (None, "final-rt"):
            raise ValueError(f"--final-rt conflicts with --clock {args.clock}")
        args.clock = "final-rt"
    if args.real_time:
        if args.clock not in (None, "realtime"):
            raise ValueError(f"--real-time conflicts with --clock {args.clock}")
        args.clock = "realtime"
    if args.clock:
        suffix = {"paused": "paused", "realtime": "realtime", "final-rt": "final_rt"}[
            args.clock
        ]
        suite = replace(
            suite,
            name=f"{suite.name}_{suffix}",
            config={
                **suite.config,
                "clock_mode": args.clock,
                "real_time": args.clock == "realtime",
            },
        )
    run_overrides = build_run_overrides(suite.config)

    max_parallel = max(1, min(args.max_parallel or len(suite.runs), len(suite.runs)))
    effective_parallel = min(max_parallel, max(len(wave) for wave in suite.repeat_waves))
    output_dir = results_dir / f"{suite.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(
        f"Suite: {suite.name}\n"
        f"Expanded runs: {len(suite.runs)}\n"
        f"Repeat waves: {len(suite.repeat_waves)}\n"
        f"Parallel workers: {effective_parallel}\n"
        f"Run overrides: {json.dumps(run_overrides, ensure_ascii=False, sort_keys=True)}\n"
        f"Base port: {args.port}\n"
        f"Results dir: {output_dir}"
    )


    context = build_suite_context(
        stamp=datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
        root=root,
        output_dir=output_dir,
        suite=suite,
        run_overrides=run_overrides,
        base_port=args.port,
        max_parallel=effective_parallel,
    )

    start_suite(context)
    started_at = datetime.now().isoformat()
    live_monitor = LiveProgressMonitor()

    rows = []
    for wave_idx, wave_runs in enumerate(suite.repeat_waves, start=1):
        rows.extend(
            run_wave(
                wave_runs,
                context=context,
                wave_idx=wave_idx,
                live_monitor=live_monitor,
                completed_rows=rows,
            )
        )

    # live_monitor.clear()
    rows.sort(key=lambda row: int(row["run_index"]))
    write_suite_outputs(output_dir, suite.name, suite.path, started_at, rows)
    update_live_suite_manifest(context, rows=rows, active_run_ids=[], final=True)
    

    print(
        "\nSuite completed.",
        f"Summary JSON: {output_dir / 'summary.json'}",
        f"Runs CSV: {output_dir / 'runs.csv'}",
        f"Aggregate by model CSV: {output_dir / 'aggregate_by_model.csv'}",
    )


if __name__ == "__main__":
    main()
