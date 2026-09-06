"""Complete command-line workflow. Run `fgpvdta --help` for all stages."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from fgpvdta.experiments.presets import SEEDS, SUITES, variants


def build_parser():
    parser = argparse.ArgumentParser(prog="fgpvdta")
    commands = parser.add_subparsers(dest="command", required=True)
    runtime = commands.add_parser(
        "configure-runtime", help="manually register a pinned environment"
    )
    runtime.add_argument("--methods", nargs="+", choices=("fg", "pv"), required=True)
    runtime.add_argument("--output", required=True, help="local profile file; do not commit it")
    download = commands.add_parser("download", help="download original DeepDTA DAVIS/KIBA data")
    download.add_argument("--output", required=True)
    download.add_argument("--revision", default="master")
    canonical = commands.add_parser(
        "canonicalize", help="convert DeepDTA matrices into validated CSV"
    )
    canonical.add_argument("--dataset", choices=("davis", "kiba"), required=True)
    canonical.add_argument("--input", required=True)
    canonical.add_argument("--output", required=True)
    fasta = commands.add_parser("export-fasta", help="CSV to one FASTA per target")
    fasta.add_argument("--csv", required=True)
    fasta.add_argument("--output", required=True)
    align = commands.add_parser("align", help="run HHblits and convert A3M to ALN")
    align.add_argument("--fasta", required=True)
    align.add_argument("--database", required=True)
    align.add_argument("--executable", default="hhblits")
    align.add_argument("--threads", type=int, default=4)
    align.add_argument("--output", required=True)
    contacts = commands.add_parser(
        "predict-contacts", help="run PconsC4 in its external environment"
    )
    contacts.add_argument("--alignments", required=True)
    contacts.add_argument(
        "--python", required=True, help="Python executable with PconsC4 installed"
    )
    contacts.add_argument("--output", required=True)
    prepare = commands.add_parser("prepare", help="build all graph features and contact images")
    prepare.add_argument("--csv", required=True)
    prepare.add_argument("--alignments", required=True)
    prepare.add_argument("--contacts", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--allow-missing", action="store_true")
    prepare.add_argument("--control-seed", type=int, default=42)
    prepare.add_argument(
        "--legacy-kiba-subset",
        action="store_true",
        help="keep 1st/3rd/5th structurally valid pairs, capped at 40000",
    )
    verify = commands.add_parser("verify", help="verify prepared artifact checksums and schema")
    verify.add_argument("--data", required=True)
    for name in ("suite", "train"):
        run = commands.add_parser(
            name, help="run reviewer suites" if name == "suite" else "train a main method"
        )
        run.add_argument("--data", required=True, help="prepared dataset directory")
        run.add_argument("--output", default="results")
        run.add_argument("--runtime-profile", help="required for execution, not for --plan")
        if name == "suite":
            run.add_argument("--name", choices=(*SUITES, "all"), default="methods")
        else:
            run.add_argument(
                "--model", choices=("dgraphdta", "fggraphdta", "pvgraphdta"), required=True
            )
        run.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
        run.add_argument("--epochs", type=int, default=250)
        run.add_argument("--batch-size", type=int, default=128)
        run.add_argument("--pv-batch-size", type=int, default=32)
        run.add_argument("--learning-rate", type=float, default=0.001)
        run.add_argument("--dropout", type=float, default=0.2)
        run.add_argument("--weight-decay", type=float, default=0.0)
        run.add_argument("--split-seed", type=int, default=42)
        run.add_argument("--test-fraction", type=float, default=0.1)
        run.add_argument("--validation-fraction", type=float, default=0.1)
        run.add_argument("--patience", type=int, default=30)
        run.add_argument("--ae-epochs", type=int, default=200)
        run.add_argument("--ae-batch-size", type=int, default=8)
        run.add_argument("--ae-seed", type=int, default=42)
        run.add_argument("--device", default="auto")
        run.add_argument("--num-workers", type=int, default=0)
        run.add_argument("--no-pretrained", action="store_true")
        run.add_argument(
            "--cold-target", action="store_true", help="make any suite protein-disjoint"
        )
        run.add_argument("--plan", action="store_true", help="print exact variants without running")
    count = commands.add_parser(
        "parameters", help="count real model parameters without downloading weights"
    )
    count.add_argument("--suite", choices=SUITES, default="methods")
    count.add_argument("--output", default="results/parameters.json")
    predict = commands.add_parser(
        "predict", help="evaluate a saved best checkpoint on its test split"
    )
    predict.add_argument("--run", required=True)
    predict.add_argument("--data", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--runtime-profile", required=True)
    stats = commands.add_parser("summarize", help="tables, seed tests and paired per-target tests")
    stats.add_argument("--results", required=True)
    stats.add_argument("--output", required=True)
    stats.add_argument("--baseline")
    stats.add_argument("--bootstrap-samples", type=int, default=10000)
    stats.add_argument("--expected-seeds", nargs="+", type=int, default=list(SEEDS))
    smoke = commands.add_parser("smoke", help="synthetic end-to-end CPU check; never paper results")
    smoke.add_argument("--output", required=True)
    smoke.add_argument(
        "--include-pv", action="store_true", help="also fit AE and a ResNet18 PV smoke model"
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "configure-runtime":
        from fgpvdta.runtime import configure_runtime

        print(f"Runtime profile created: {configure_runtime(args.output, args.methods)}")
    elif args.command == "download":
        from fgpvdta.data.download import download_deepdta_data

        download_deepdta_data(args.output, args.revision)
    elif args.command == "canonicalize":
        from fgpvdta.data.deepdta import canonicalize_deepdta_dataset
        from fgpvdta.data.schema import read_interactions

        if Path(args.output).exists():
            raise FileExistsError(args.output)
        canonicalize_deepdta_dataset(args.input, args.dataset, args.output)
        read_interactions(args.output)
    elif args.command == "export-fasta":
        from fgpvdta.preprocessing.external import export_fasta

        export_fasta(args.csv, args.output)
    elif args.command == "align":
        from fgpvdta.preprocessing.external import run_hhblits

        run_hhblits(args.fasta, args.output, args.database, args.executable, args.threads)
    elif args.command == "predict-contacts":
        from fgpvdta.preprocessing.external import predict_contacts

        predict_contacts(args.alignments, args.output, args.python)
    elif args.command == "prepare":
        from fgpvdta.preprocessing.pipeline import prepare

        prepare(
            args.csv,
            args.alignments,
            args.contacts,
            args.output,
            args.allow_missing,
            args.control_seed,
            args.legacy_kiba_subset,
        )
    elif args.command == "verify":
        from fgpvdta.preprocessing.pipeline import verify_prepared

        frame, _ = verify_prepared(args.data)
        print(f"Verified {len(frame)} interactions, {frame.Target_ID.nunique()} targets")
    elif args.command in {"suite", "train"}:
        from fgpvdta.runtime import require_runtime

        suites = (
            (SUITES if args.name == "all" else (args.name,))
            if args.command == "suite"
            else ("methods",)
        )
        selections = {
            suite: [v for v in variants(suite) if args.command == "suite" or v.name == args.model]
            for suite in suites
        }
        if not args.plan:
            require_runtime(
                args.runtime_profile,
                {v.family for selected in selections.values() for v in selected},
            )
        from fgpvdta.experiments.runner import RunOptions, run_suite

        options = RunOptions(
            **{k: v for k, v in vars(args).items() if k in RunOptions.__dataclass_fields__}
        )
        options = replace(
            options, seeds=tuple(args.seeds), pretrained_resnet=not args.no_pretrained
        )
        for suite, selected in selections.items():
            if args.plan:
                print(
                    json.dumps(
                        {
                            "suite": suite,
                            "options": asdict(options),
                            "variants": [asdict(v) for v in selected],
                        },
                        indent=2,
                    )
                )
            else:
                run_suite(suite, options, selected)
    elif args.command == "parameters":
        from fgpvdta.data.schema import write_json
        from fgpvdta.experiments.runner import parameter_counts

        write_json(args.output, [parameter_counts(v) for v in variants(args.suite)])
    elif args.command == "predict":
        from fgpvdta.runtime import require_runtime

        spec = json.loads((Path(args.run) / "configuration.json").read_text())
        require_runtime(args.runtime_profile, {spec["variant"]["family"]})
        from fgpvdta.experiments.runner import predict_checkpoint

        predict_checkpoint(args.run, args.data, args.output, args.device)
    elif args.command == "summarize":
        from fgpvdta.analysis.reviewer import summarize

        summarize(
            args.results, args.output, args.baseline, args.bootstrap_samples, args.expected_seeds
        )
    elif args.command == "smoke":
        from fgpvdta.experiments.smoke import run_smoke

        run_smoke(args.output, args.include_pv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
