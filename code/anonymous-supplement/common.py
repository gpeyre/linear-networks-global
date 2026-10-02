"""Headless plotting and identity-free output metadata."""
from pathlib import Path
import hashlib
import json
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "supplement-mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#2458A6", "#D1495B", "#00876C", "#7A3E9D", "#E08D2D", "#2B8CBE"]
GRID = {"which": "both", "color": "#E8E8E8", "linewidth": 0.8}


def configure():
    plt.rcdefaults()
    plt.rcParams.update({
        "font.size": 6, "font.family": "serif", "axes.titlesize": 8,
        "axes.labelsize": 6, "legend.fontsize": 6, "figure.dpi": 140,
        "savefig.dpi": 200, "pdf.fonttype": 42, "ps.fonttype": 42,
        "text.usetex": False, "pgf.texsystem": "pdflatex", "pgf.rcfonts": False,
        "pgf.preamble": r"\usepackage{amsfonts,amssymb}",
    })


def save_figure(fig, output, stem, formats):
    directory = Path(output) / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in formats:
        metadata = None
        if suffix == "pdf":
            metadata = {"Author": None, "Creator": "Anonymous supplementary numerics",
                        "CreationDate": None, "ModDate": None}
        options = {"metadata": metadata} if suffix == "pdf" else {}
        fig.savefig(directory / f"{stem}.{suffix}", bbox_inches="tight", pad_inches=0.04,
                    **options)
    plt.close(fig)


def save_data(output, stem, **arrays):
    directory = Path(output) / "data"
    directory.mkdir(parents=True, exist_ok=True)
    if any(np.asarray(value).dtype.hasobject for value in arrays.values()):
        raise ValueError("Only numeric arrays may be cached")
    np.savez_compressed(directory / f"{stem}.npz", **arrays)


def load_data(directory, stem):
    with np.load(Path(directory) / f"{stem}.npz", allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def validate_arrays(data, shapes):
    """Reject incomplete, non-numeric, non-finite, or mis-shaped cached runs."""
    missing = set(shapes) - set(data)
    if missing:
        raise ValueError(f"Missing data arrays: {', '.join(sorted(missing))}")
    for name, value in data.items():
        array = np.asarray(value)
        if array.dtype.kind not in "biuf" or not np.all(np.isfinite(array)):
            raise ValueError(f"Non-finite or non-numeric data: {name}")
        if name in shapes and array.shape != shapes[name]:
            raise ValueError(f"Wrong shape for {name}: {array.shape}, expected {shapes[name]}")


def check_times(times):
    times = np.asarray(times)
    if (times.ndim != 1 or len(times) < 2 or not np.all(np.isfinite(times))
            or times[0] != 0 or not np.all(np.diff(times) > 0)):
        raise ValueError("Times must be finite, strictly increasing, and start at zero")


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
