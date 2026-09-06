"""Shared training, immutable splits, leakage-safe AE fitting, and observed outputs."""

import gc
import json
import platform
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pandas as pd
import torch

from fgpvdta.data.download import sha256_file
from fgpvdta.data.schema import (
    fingerprint,
    frame_fingerprint,
    implementation_fingerprint,
    write_json,
)
from fgpvdta.data.splits import (
    DatasetSplit,
    assert_no_target_leakage,
    make_cold_target_split,
    make_random_split,
    validate_split_coverage,
)
from fgpvdta.models.autoencoder import (
    ContactMapAutoencoder,
    export_autoencoder_latents,
    train_autoencoder,
)
from fgpvdta.models.fggraphdta import FGGraphDTA
from fgpvdta.models.pvgraphdta import PVGraphDTA
from fgpvdta.preprocessing.pipeline import verify_prepared
from fgpvdta.seed import seed_everything
from fgpvdta.training.datasets import InteractionDataset, build_interaction_loader
from fgpvdta.training.engine import TrainingConfig, train_model
from fgpvdta.training.inference import export_predictions, predict_loader
from fgpvdta.training.metrics import regression_metrics
from fgpvdta.training.profiling import count_parameters

from .presets import SEEDS, Variant, variants


@dataclass(frozen=True)
class RunOptions:
    data: str
    output: str = "results"
    seeds: tuple[int, ...] = SEEDS
    epochs: int = 250
    batch_size: int = 128
    pv_batch_size: int = 32
    learning_rate: float = 0.001
    dropout: float = 0.2
    weight_decay: float = 0.0
    split_seed: int = 42
    test_fraction: float = 0.1
    validation_fraction: float = 0.1
    patience: int = 30
    ae_epochs: int = 200
    ae_batch_size: int = 8
    ae_seed: int = 42
    pretrained_resnet: bool = True
    device: str = "auto"
    num_workers: int = 0
    synthetic: bool = False
    cold_target: bool = False

    def __post_init__(self):
        if not 0 < self.test_fraction < 1 or not 0 < self.validation_fraction < 1:
            raise ValueError("Reviewer runs require positive validation and test fractions")
        if self.test_fraction + self.validation_fraction >= 1:
            raise ValueError("Fractions leave no training samples")
        if (
            min(
                self.epochs, self.batch_size, self.pv_batch_size, self.ae_epochs, self.ae_batch_size
            )
            < 1
        ):
            raise ValueError("Epochs and batch sizes must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid optimizer/dropout configuration")
        if min(self.split_seed, self.ae_seed, self.patience, self.num_workers) < 0:
            raise ValueError("Seeds, patience and worker count must be nonnegative")


def make_model(variant, dropout=0.2, pretrained=True):
    if variant.family == "fg":
        return FGGraphDTA(variant.fg_variant, dropout=dropout, fg_count=variant.fg_count)
    return PVGraphDTA(
        dropout=dropout,
        use_vision=variant.vision,
        use_ae_gating=variant.ae_gate,
        use_infonce=variant.infonce,
        resnet_variant=variant.resnet,
        pretrained_resnet=pretrained,
        temperature=variant.tau,
    )


def parameter_counts(variant):
    model = make_model(variant, pretrained=False)
    return {
        "variant": variant.name,
        "trainable": count_parameters(model),
        "total": count_parameters(model, False),
    }


def frozen_split(frame, root, kind, options):
    specification = {
        "dataset_sha256": frame_fingerprint(frame),
        "kind": kind,
        "seed": options.split_seed,
        "test_fraction": options.test_fraction,
        "validation_fraction": options.validation_fraction,
        "target_grouping": "exact_sequence" if kind == "cold_target" else "row",
    }
    path = Path(root) / "splits" / f"{kind}_{fingerprint(specification)[:16]}.json"
    if path.exists():
        payload = json.loads(path.read_text())
        if payload["specification"] != specification:
            raise ValueError("Stored split specification mismatch")
        split = DatasetSplit(**payload["split"])
    else:
        if kind == "cold_target":
            # Identical sequences under different IDs must not cross partitions.
            split = make_cold_target_split(
                frame,
                "Target",
                options.test_fraction,
                options.validation_fraction,
                options.split_seed,
            )
        else:
            split = make_random_split(
                frame, options.test_fraction, options.validation_fraction, options.split_seed
            )
        payload = {"specification": specification, "split": asdict(split)}
        write_json(path, payload)
    validate_split_coverage(frame, split)
    if options.validation_fraction > 0 and not split.validation:
        raise ValueError("Requested validation partition is empty")
    if kind == "cold_target":
        assert_no_target_leakage(frame, split, "Target_ID")
        assert_no_target_leakage(frame, split, "Target")
    return split, path, fingerprint(payload)


def split_autoencoder(frame, split, split_hash, data, output, options, device):
    """Fit exclusively on train targets, then freeze and encode all targets."""
    target_ids = sorted(frame.iloc[split.train].Target_ID.unique())
    spec = {
        "split_sha256": split_hash,
        "ae_implementation_sha256": sha256_file(
            Path(__file__).parents[1] / "models" / "autoencoder.py"
        ),
        "training_target_ids": target_ids,
        "epochs": options.ae_epochs,
        "seed": options.ae_seed,
        "batch_size": options.ae_batch_size,
        "prepared_sha256": sha256_file(data / "manifest.json"),
    }
    root = output / "autoencoders" / fingerprint(spec)[:16]
    manifest = root / "manifest.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text())
        if payload["specification"] != spec:
            raise ValueError("AE training provenance mismatch")
        for name, expected in payload["files"].items():
            if sha256_file(root / name) != expected:
                raise ValueError(f"AE artifact changed: {name}")
        return root / "latents", payload
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Incomplete AE run: use a new output directory: {root}")
    start = time.perf_counter()
    checkpoint = train_autoencoder(
        data / "images" / "native",
        root,
        epochs=options.ae_epochs,
        batch_size=options.ae_batch_size,
        seed=options.ae_seed,
        device=device,
        num_workers=options.num_workers,
        target_ids=target_ids,
    )
    model = ContactMapAutoencoder()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    export_autoencoder_latents(
        model,
        data / "images" / "native",
        root / "latents",
        batch_size=options.ae_batch_size,
        num_workers=options.num_workers,
        device=device,
    )
    payload = {
        "specification": spec,
        "fit_and_export_seconds": time.perf_counter() - start,
        "files": {str(p.relative_to(root)): sha256_file(p) for p in root.rglob("*.pt")},
    }
    write_json(manifest, payload)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return root / "latents", payload


def _loaders(frame, split, data, variant, options, seed, latents):
    ligand_key = protein_key = "k0"
    if variant.family == "fg":
        feature_key = "random_k20" if variant.random_fg else f"k{variant.fg_count}"
        if variant.fg_variant in {"ligand_fg", "full"}:
            ligand_key = feature_key
        if variant.fg_variant in {"protein_fg", "full"}:
            protein_key = feature_key
    batch_size = options.pv_batch_size if variant.family == "pv" else options.batch_size
    loaders = []
    for name, indices in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        if not indices:
            loaders.append(None)
            continue
        dataset = InteractionDataset(
            frame,
            indices,
            data / "ligands" / ligand_key,
            data / "proteins" / protein_key,
            image_dir=data / "images" / "resize256",
            ae_latent_dir=latents,
            require_vision=variant.vision,
            require_ae=variant.ae_gate,
        )
        loaders.append(
            build_interaction_loader(
                dataset, batch_size, name == "train", seed, options.num_workers
            )
        )
    return loaders


def run_suite(suite, options, selected_variants=None):
    if not options.seeds or len(options.seeds) != len(set(options.seeds)):
        raise ValueError("A nonempty list of unique seeds is required")
    data, output = Path(options.data).resolve(), Path(options.output).resolve()
    options = replace(options, data=str(data), output=str(output))
    frame, prepared = verify_prepared(data)
    code_hash = implementation_fingerprint()
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if options.device == "auto"
        else options.device
    )
    chosen = variants(suite) if selected_variants is None else selected_variants
    kind = "cold_target" if suite == "cold_target" or options.cold_target else "random"
    experiment_name = suite + (
        "_cold_target" if options.cold_target and suite != "cold_target" else ""
    )
    split, split_path, split_hash = frozen_split(frame, output, kind, options)
    latents = ae_provenance = None
    if any(v.ae_gate for v in chosen):
        latents, ae_provenance = split_autoencoder(
            frame, split, split_hash, data, output, options, device
        )
    results = []
    for variant in chosen:
        for seed in options.seeds:
            root = output / experiment_name / variant.name / f"seed_{seed}"
            result_path = root / "result.json"
            spec = {
                "implementation_sha256": code_hash,
                "options": asdict(options),
                "variant": asdict(variant),
                "seed": seed,
                "split_sha256": split_hash,
                "prepared_sha256": fingerprint(prepared),
                "ae_provenance": ae_provenance if variant.ae_gate else None,
            }
            spec_hash = fingerprint(spec)
            if result_path.exists():
                old = json.loads(result_path.read_text())
                if old.get("run_sha256") != spec_hash:
                    raise ValueError(f"Changed run configuration at {root}; use a new --output")
                if sha256_file(root / "test_predictions.csv") != old["predictions_sha256"]:
                    raise ValueError("Saved predictions changed")
                if sha256_file(root / "checkpoints" / "best.pt") != old["checkpoint_sha256"]:
                    raise ValueError("Saved checkpoint changed")
                results.append(result_path)
                continue
            if root.exists() and any(root.iterdir()):
                raise FileExistsError(f"Incomplete run: {root}; use a new --output")
            root.mkdir(parents=True, exist_ok=True)
            # Seed before construction, not just at the start of training.
            seed_everything(seed)
            model = make_model(variant, options.dropout, options.pretrained_resnet)
            train, validation, test = _loaders(frame, split, data, variant, options, seed, latents)
            if validation is None:
                raise ValueError("Reviewer runs require a validation partition")
            training = TrainingConfig(
                epochs=options.epochs,
                learning_rate=options.learning_rate,
                weight_decay=options.weight_decay,
                alpha=variant.alpha,
                seed=seed,
                early_stopping_patience=options.patience,
                device=device,
            )
            write_json(root / "configuration.json", spec)
            trained = train_model(
                model,
                variant.family,
                train,
                training,
                root / "checkpoints",
                validation_loader=validation,
                metadata=spec,
            )
            predictions = predict_loader(model, test, variant.family, device)
            metrics = export_predictions(
                predictions, root / "test_predictions.csv", root / "test_metrics.json"
            )
            per_target = []
            for target_id, group in predictions.groupby("Target_ID"):
                per_target.append(
                    {
                        "Target_ID": target_id,
                        **regression_metrics(group.y_true, group.y_pred).to_dict(),
                    }
                )
            pd.DataFrame(per_target).to_csv(root / "per_target_metrics.csv", index=False)
            result = {
                "schema": "fgpvdta.run.v2",
                "observed": not options.synthetic,
                "synthetic": options.synthetic,
                "experiment": experiment_name,
                "variant": variant.name,
                "split_kind": kind,
                "seed": seed,
                "run_sha256": spec_hash,
                "implementation_sha256": code_hash,
                "split_sha256": split_hash,
                "split_path": str(split_path),
                "dataset_sha256": frame_fingerprint(frame),
                "test_metrics": metrics,
                "selected_epoch": trained.selected_epoch,
                "total_runtime_seconds": trained.total_runtime_seconds,
                "ae_fit_and_export_seconds": ae_provenance.get("fit_and_export_seconds")
                if variant.ae_gate
                else None,
                "peak_gpu_memory_mb": trained.peak_gpu_memory_mb,
                "trainable_parameters": trained.trainable_parameters,
                "total_parameters": trained.total_parameters,
                "predictions_csv": "test_predictions.csv",
                "predictions_sha256": sha256_file(root / "test_predictions.csv"),
                "checkpoint_sha256": sha256_file(root / "checkpoints" / "best.pt"),
                "environment": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "device": device,
                },
            }
            write_json(result_path, result)
            results.append(result_path)
            del model, train, validation, test
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return results


def predict_checkpoint(run_directory, data, output_csv, device="cpu"):
    """Re-run the selected checkpoint on its stored test partition only."""
    root, data = Path(run_directory), Path(data).resolve()
    spec = json.loads((root / "configuration.json").read_text())
    frame, prepared = verify_prepared(data)
    if fingerprint(prepared) != spec["prepared_sha256"]:
        raise ValueError("Prediction dataset differs from training")
    options = RunOptions(**spec["options"])
    variant = Variant(**spec["variant"])
    kind = json.loads((root / "result.json").read_text())["split_kind"]
    split, _, split_hash = frozen_split(frame, options.output, kind, options)
    if split_hash != spec["split_sha256"]:
        raise ValueError("Prediction split mismatch")
    latents = None
    if variant.ae_gate:
        ae_spec = spec["ae_provenance"]["specification"]
        latents = Path(options.output) / "autoencoders" / fingerprint(ae_spec)[:16] / "latents"
    _, _, test = _loaders(frame, split, data, variant, options, spec["seed"], latents)
    model = make_model(variant, options.dropout, pretrained=False)
    payload = torch.load(root / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    predictions = predict_loader(model, test, variant.family, device)
    return export_predictions(
        predictions, output_csv, Path(output_csv).with_suffix(".metrics.json")
    )
