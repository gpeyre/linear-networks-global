"""Recompute the six ICLR article figures from synthetic initializations."""
import argparse
import importlib
from pathlib import Path
import platform
import time
import unittest

from common import configure, file_digest, load_data, write_json
import matplotlib
import numpy as np
import scipy
import threadpoolctl


FIGURES = {
    1: ("geometry", "render", "figure1_covariance_cone"),
    2: ("first_order", None, "figure2_training"),
    3: ("certificates", "windows", "figure3_damping_windows"),
    4: ("momentum", None, "figure4_momentum"),
    5: ("certificates", "rate_plot", "figure5_boundary_rates"),
    6: ("smooth_lasso", None, "figure6_smooth_lasso"),
}


def run_checks():
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError("Figure consistency checks failed")
    return {"tests_run": result.testsRun, "failures": 0, "errors": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", nargs="+", type=int, choices=FIGURES,
                        default=list(FIGURES), metavar="N", help="Figure numbers, default: all six")
    parser.add_argument("--output", type=Path, default=Path("outputs"),
                        help="Output directory, relative to the working directory")
    parser.add_argument("--formats", nargs="+", choices=["pdf", "png", "pgf"], default=["pdf"],
                        help="PGF additionally requires pdflatex; PDF and PNG do not")
    parser.add_argument("--from-data", type=Path, metavar="DIRECTORY",
                        help="Replot Figures 2, 4, 6 from a previous run's data directory")
    parser.add_argument("--check", action="store_true", help="Also run the focused unit tests")
    args = parser.parse_args()
    if not __debug__:
        parser.error("Do not use Python -O: numerical validation uses assertions")
    numbers = list(dict.fromkeys(args.figures))
    formats = list(dict.fromkeys(args.formats))
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "verification.json"
    report = {
        "status": "running",
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
                        "threadpoolctl": threadpoolctl.__version__, "blas_threads": 1},
        "requested_figures": numbers, "formats": formats,
        "mode": "replot previous data" if args.from_data else "full recomputation",
        "figures": {}, "outputs_sha256": {},
    }
    # Record status first so a failed rerun cannot leave an old success report.
    write_json(report_path, report)
    start_run = time.perf_counter()
    try:
        with threadpoolctl.threadpool_limits(limits=1):
            if args.check:
                report["unit_tests"] = run_checks()
            for number in numbers:
                module_name, render_name, stem = FIGURES[number]
                print(f"Figure {number}: {stem}", flush=True)
                configure()
                module = importlib.import_module(module_name)
                start = time.perf_counter()
                if render_name is not None:
                    diagnostics = getattr(module, render_name)(args.output, formats)
                else:
                    if args.from_data is None:
                        data, diagnostics = module.compute(args.output)
                    else:
                        data = load_data(args.from_data, stem)
                        diagnostics = module.validate(data)
                    module.plot(data, args.output, formats)
                diagnostics["seconds"] = round(time.perf_counter() - start, 3)
                report["figures"][str(number)] = diagnostics
                for suffix in formats:
                    path = args.output / "figures" / f"{stem}.{suffix}"
                    report["outputs_sha256"][f"figures/{path.name}"] = file_digest(path)
                if render_name is None and args.from_data is None:
                    path = args.output / "data" / f"{stem}.npz"
                    report["outputs_sha256"][f"data/{path.name}"] = file_digest(path)
                write_json(report_path, report)
    except Exception:
        report["status"] = "failed"
        write_json(report_path, report)
        raise
    report["status"] = "passed"
    report["total_seconds"] = round(time.perf_counter() - start_run, 3)
    write_json(report_path, report)
    print("All requested figures and validation checks completed.", flush=True)


if __name__ == "__main__":
    main()
