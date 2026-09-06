"""External HHblits/PconsC4 pipeline; executable/database paths are explicit."""

import subprocess
from pathlib import Path

from fgpvdta.data.schema import read_interactions


def export_fasta(csv, output):
    frame = read_interactions(csv)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for row in frame[["Target_ID", "Target"]].drop_duplicates().itertuples(index=False):
        path = output / f"{row.Target_ID}.fasta"
        text = (
            f">{row.Target_ID}\n"
            + "\n".join(row.Target[i : i + 60] for i in range(0, len(row.Target), 60))
            + "\n"
        )
        if path.exists() and path.read_text() != text:
            raise FileExistsError(path)
        path.write_text(text, encoding="utf-8")


def a3m_to_aln(source, destination):
    records = []
    current = ""
    for line in Path(source).read_text().splitlines():
        if line.startswith(">"):
            if current:
                records.append(current)
            current = ""
        elif not line.startswith("#"):
            current += "".join(c for c in line.strip() if not c.islower() and c != ".")
    if current:
        records.append(current)
    if not records or any(len(r) != len(records[0]) for r in records):
        raise ValueError("A3M does not produce a consistent aligned query length")
    Path(destination).write_text("\n".join(records) + "\n", encoding="utf-8")


def run_hhblits(fasta_dir, output, database, executable="hhblits", threads=4):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(fasta_dir).glob("*.fasta"))
    if not files:
        raise FileNotFoundError("No FASTA files; run export-fasta first")
    for fasta in files:
        a3m = output / f"{fasta.stem}.a3m"
        if a3m.exists():
            raise FileExistsError(a3m)
        subprocess.run(
            [
                executable,
                "-i",
                str(fasta),
                "-d",
                str(database),
                "-oa3m",
                str(a3m),
                "-n",
                "3",
                "-cpu",
                str(threads),
            ],
            check=True,
        )
        a3m_to_aln(a3m, output / f"{fasta.stem}.aln")


def predict_contacts(alignment_dir, output, python_executable):
    """Run each target in a separate legacy Python environment, as in the source."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    worker = Path(__file__).with_name("pconsc4_worker.py")
    files = sorted(Path(alignment_dir).glob("*.aln"))
    if not files:
        raise FileNotFoundError("No .aln alignments")
    for alignment in files:
        destination = output / f"{alignment.stem}.npy"
        if destination.exists():
            raise FileExistsError(destination)
        subprocess.run(
            [str(python_executable), str(worker), str(alignment), str(destination)], check=True
        )
