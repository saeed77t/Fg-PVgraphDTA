"""Inventory notebook sources without copying notebooks or executing their cells."""

import argparse
import ast
import hashlib
import json
from pathlib import Path


def inventory(source):
    records = []
    for path in sorted(source.glob("*.ipynb")):
        raw = path.read_bytes()
        notebook = json.loads(raw)
        cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
        definitions = []
        for index, cell in enumerate(cells):
            try:
                tree = ast.parse(cell)
            except SyntaxError:
                continue  # Magics are inspected as text; never executed.
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    definitions.append({"name": node.name, "code_cell": index + 1})
        records.append(
            {
                "file": path.name,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "code_cells": len(cells),
                "definitions": definitions,
            }
        )
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("docs/source_inventory.json"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory(args.source), indent=2) + "\n", encoding="utf-8")
