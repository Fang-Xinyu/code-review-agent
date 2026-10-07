from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


EXCLUDED_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache", "dist", "build", "reports"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".log", ".zip"}


def allowed(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if any(part in EXCLUDED_DIRS for part in relative.parts):
        return False
    if path.suffix in EXCLUDED_SUFFIXES:
        return False
    if path.name == ".env" or (path.name.startswith(".env.") and path.name != ".env.example"):
        return False
    if path.name == ".DS_Store":
        return False
    if path.name.startswith("report") and path.suffix in {".json", ".md"}:
        return False
    return True


def package(root: Path, output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    files = [p for p in root.rglob("*") if p.is_file() and allowed(p, root) and p.resolve() != output.resolve()]
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, Path(root.name) / path.relative_to(root))
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="输出路径；默认在项目文件夹旁生成同名 ZIP，相对路径以项目目录为基准")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (root / args.output).resolve() if args.output else root.parent / f"{root.name}.zip"
    count = package(root, output)
    print(f"Packaged {count} files.\nSaved to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
