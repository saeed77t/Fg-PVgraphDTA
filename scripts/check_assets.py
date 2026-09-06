"""Check a bounded sample of real CSV/alignment/contact assets without training."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fgpvdta.data.schema import write_json
from fgpvdta.preprocessing.contact_images import build_historical_stacked_rgb
from fgpvdta.preprocessing.contact_maps import load_contact_map
from fgpvdta.preprocessing.ligand_graphs import build_ligand_graph
from fgpvdta.preprocessing.protein_graphs import build_protein_graph, find_contact_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--alignments", type=Path, required=True)
    parser.add_argument("--contacts", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    frame = pd.read_csv(args.csv, nrows=args.limit, dtype={"Drug_ID": str, "Target_ID": str})
    report = []
    for row in frame.itertuples(index=False):
        ligand = build_ligand_graph(row.Drug, True)
        cm = find_contact_map(args.contacts, row.Target_ID)
        protein = build_protein_graph(
            row.Target_ID, row.Target, args.alignments / f"{row.Target_ID}.aln", cm, True
        )
        image = build_historical_stacked_rgb(
            load_contact_map(cm, len(row.Target), "none"), row.Target
        )
        assert ligand.x.shape[1] == 98 and protein.x.shape[1] == 74
        assert np.isfinite(image).all() and not image[..., 2].any()
        report.append(
            {
                "drug_id": row.Drug_ID,
                "target_id": row.Target_ID,
                "ligand_shape": list(ligand.x.shape),
                "protein_shape": list(protein.x.shape),
                "rgb_shape": list(image.shape),
                "passed": True,
            }
        )
    write_json(
        args.output,
        {"scope": "real preprocessing only; no training or affinity results", "checks": report},
    )


if __name__ == "__main__":
    main()
