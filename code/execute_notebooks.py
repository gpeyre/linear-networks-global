"""Execute example notebooks in fresh, isolated kernels using this interpreter."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/notebooks"))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    examples = Path(__file__).resolve().parent / "notebooks" / "examples"
    paths = sorted(examples.glob("*.ipynb"))
    if not paths:
        raise FileNotFoundError(f"No example notebooks in {examples}")
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
        with tempfile.TemporaryDirectory(prefix="linear-networks-notebook-") as tmp:
            root = Path(tmp)
            spec = root / "kernels" / "toolbox"
            spec.mkdir(parents=True)
            (spec / "kernel.json").write_text(json.dumps({
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "Toolbox validation", "language": "python",
            }))
            manager = KernelManager(kernel_name="toolbox", kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(spec.parent)]))
            client = NotebookClient(notebook, km=manager, timeout=180,
                                    resources={"metadata": {"path": str(root)}})
            try:
                client.execute(env={**os.environ, "MPLCONFIGDIR": str(root / "matplotlib")})
            finally:
                if manager.has_kernel:
                    manager.shutdown_kernel(now=True)
                if client.kc is not None:
                    client.kc.stop_channels()
        for cell in notebook.cells:
            cell.metadata.pop("execution", None)
        nbformat.write(notebook, output / path.name)
        print(f"Executed {path.name}", flush=True)


if __name__ == "__main__":
    main()
