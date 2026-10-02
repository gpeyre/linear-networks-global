"""Build a deterministic source-only archive using an explicit file allowlist."""
import argparse
import hashlib
from pathlib import Path
import re
import stat
import zipfile


FILES = (
    "README.md", "EXPERIMENTS.md", "requirements.txt", "requirements-tested.txt",
    "reproduce.py", "common.py", "geometry.py", "first_order.py", "certificates.py",
    "momentum.py", "smooth_lasso.py", "tests/test_figures.py", "package.py",
)


def build(destination):
    root = Path(__file__).resolve().parent
    destination = Path(destination).resolve()
    if destination in {root / name for name in FILES}:
        raise ValueError("The archive destination must not overwrite a source file")
    payload = {}
    for name in FILES:
        source = root / name
        if source.is_symlink():
            raise ValueError(f"Refusing symbolic link: {name}")
        content = source.read_bytes()
        text = content.decode("utf-8")
        if re.search(r"/(?:Users|home)/\S+|[A-Za-z]:\\Users\\\S+", text):
            raise ValueError(f"Personal filesystem path in {name}")
        if re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text):
            raise ValueError(f"Email address in {name}")
        payload[name] = content
    manifest = "".join(f"{hashlib.sha256(data).hexdigest()}  {name}\n"
                       for name, data in sorted(payload.items()))
    payload["MANIFEST.sha256"] = manifest.encode("ascii")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(payload.items()):
            entry = zipfile.ZipInfo(f"supplementary/{name}", date_time=(2000, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = (stat.S_IFREG | 0o644) << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, data)
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Archive integrity check failed")
    return {"files": len(payload), "bytes": destination.stat().st_size,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent.parent / "supplementary.zip")
    args = parser.parse_args()
    result = build(args.output)
    print(f"Packaged {result['files']} files ({result['bytes']} bytes)")
    print(f"SHA-256: {result['sha256']}")
