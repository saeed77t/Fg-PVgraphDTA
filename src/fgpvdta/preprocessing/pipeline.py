"""CSV -> aligned residue graphs, ligand graphs, FG controls and contact images."""

import json
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem

from fgpvdta.data.download import sha256_file
from fgpvdta.data.schema import frame_fingerprint, read_interactions, write_json

from .contact_images import historical_pv_resize_to_256, save_historical_stacked_png
from .contact_maps import load_contact_map
from .feature_sets import feature_spec, ligand_features, protein_features, random_control
from .ligand_graphs import build_ligand_graph
from .protein_graphs import build_protein_graph, find_contact_map


def prepare(
    csv,
    alignments,
    contacts,
    output,
    allow_missing=False,
    control_seed=42,
    legacy_kiba_subset=False,
):
    frame = read_interactions(csv)
    output, alignments, contacts = Path(output), Path(alignments), Path(contacts)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    missing, sources = {}, {}
    for row in frame[["Target_ID", "Target"]].drop_duplicates().itertuples(index=False):
        try:
            aln = alignments / f"{row.Target_ID}.aln"
            if not aln.is_file():
                raise FileNotFoundError(aln)
            contact = find_contact_map(contacts, row.Target_ID)
            sources[row.Target_ID] = {
                "alignment": str(aln.resolve()),
                "alignment_sha256": sha256_file(aln),
                "contact": str(contact.resolve()),
                "contact_sha256": sha256_file(contact),
            }
        except FileNotFoundError as exc:
            missing[row.Target_ID] = str(exc)
    report = {"source_rows": len(frame), "missing_targets": missing}
    write_json(output / "filter_report.json", report)
    if missing and not allow_missing:
        raise FileNotFoundError(
            "Missing structural assets; see filter_report.json. "
            "Use --allow-missing only to explicitly define a structural subset."
        )
    frame = frame[~frame.Target_ID.isin(missing)].reset_index(drop=True)
    if legacy_kiba_subset:
        # Counter increments before testing oddness: first, third, fifth valid rows.
        frame = frame.iloc[::2].head(40000).reset_index(drop=True)
    if frame.empty:
        raise ValueError("No structurally usable interactions remain")
    frame.to_csv(output / "interactions.csv", index=False)
    for row in frame[["Drug_ID", "Drug"]].drop_duplicates().itertuples(index=False):
        base = build_ligand_graph(row.Drug, drug_id=row.Drug_ID)
        mol = Chem.MolFromSmiles(row.Drug)
        for count in (0, 10, 20, 30):
            fg = ligand_features(mol, count)
            _save_variant(base, fg, output / "ligands" / f"k{count}", row.Drug_ID)
            if count == 20:
                _save_variant(
                    base,
                    random_control(fg, "ligand:" + row.Drug_ID, control_seed),
                    output / "ligands" / "random_k20",
                    row.Drug_ID,
                )
    for row in frame[["Target_ID", "Target"]].drop_duplicates().itertuples(index=False):
        source = sources[row.Target_ID]
        base = build_protein_graph(
            row.Target_ID,
            row.Target,
            source["alignment"],
            source["contact"],
            symmetrize_contacts="none",
        )
        for count in (0, 10, 20, 30):
            fg = protein_features(row.Target, count)
            _save_variant(base, fg, output / "proteins" / f"k{count}", row.Target_ID)
            if count == 20:
                _save_variant(
                    base,
                    random_control(fg, "protein:" + row.Target_ID, control_seed),
                    output / "proteins" / "random_k20",
                    row.Target_ID,
                )
        cm = load_contact_map(source["contact"], len(row.Target), "none")
        native = output / "images" / "native" / f"{row.Target_ID}_stacked.png"
        save_historical_stacked_png(cm, row.Target, native)
        historical_pv_resize_to_256(native, output / "images" / "resize256" / native.name)
    report.update(
        retained_rows=len(frame),
        retained_targets=frame.Target_ID.nunique(),
        retained_drugs=frame.Drug_ID.nunique(),
    )
    write_json(output / "filter_report.json", report)
    manifest = {
        "schema": "fgpvdta.prepared.v1",
        "dataset_sha256": frame_fingerprint(frame),
        "source_csv_sha256": sha256_file(csv),
        "sources": sources,
        "feature_sets": {str(k): feature_spec(k) for k in (0, 10, 20, 30)},
        "random_control": "per-node FG-bit permutation preserving row sparsity",
        "control_seed": control_seed,
        "pair_selection": "first_third_fifth_valid_capped_40000" if legacy_kiba_subset else "all",
        "files": {
            str(p.relative_to(output)): sha256_file(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    }
    write_json(output / "manifest.json", manifest)
    return output


def _save_variant(base, fg, directory, entity_id):
    graph = base.clone()
    graph.x = torch.cat((base.x, torch.from_numpy(np.asarray(fg, dtype=np.float32))), dim=1)
    graph.feature_dim = graph.x.shape[1]
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(graph, directory / f"{entity_id}_graph.pt")


def verify_prepared(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Prepared artifact changed or missing: {path}")
    frame = read_interactions(root / "interactions.csv")
    if frame_fingerprint(frame) != manifest["dataset_sha256"]:
        raise ValueError("Interaction dataset fingerprint mismatch")
    return frame, manifest
