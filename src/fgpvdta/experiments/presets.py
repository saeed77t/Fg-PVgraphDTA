"""Readable, versioned Python configurations for every requested comparison."""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Variant:
    name: str
    family: str = "fg"
    fg_variant: str = "base"
    fg_count: int = 20
    random_fg: bool = False
    vision: bool = False
    ae_gate: bool = False
    infonce: bool = False
    alpha: float = 0.5
    tau: float = 0.07
    resnet: str = "resnet101"


BASELINE = Variant("dgraphdta")
FG = Variant("fggraphdta", fg_variant="full")
PV = Variant("pvgraphdta", family="pv", vision=True, ae_gate=True, infonce=True)
SEEDS = (11, 22, 33, 44, 55)


def variants(suite):
    presets = {
        "methods": [BASELINE, FG, PV],
        "fg_ablation": [
            BASELINE,
            Variant("ligand_fg", fg_variant="ligand_fg"),
            Variant("protein_fg", fg_variant="protein_fg"),
            FG,
        ],
        "fg_count": [replace(FG, name=f"fg_k{k}", fg_count=k) for k in (10, 20, 30)],
        "random_control": [BASELINE, FG, replace(FG, name="random_fg", random_fg=True)],
        "pv_ablation": [
            Variant("graph_only", family="pv"),
            replace(PV, name="graph_vision", ae_gate=False, infonce=False),
            replace(PV, name="graph_vision_ae", infonce=False),
            PV,
        ],
        "sensitivity": [PV]
        + [replace(PV, name=f"alpha_{a}", alpha=a) for a in (0.0, 0.1, 1.0)]
        + [replace(PV, name=f"tau_{t}", tau=t) for t in (0.03, 0.1, 0.2)],
        "backbone": [replace(PV, name="resnet18", resnet="resnet18"), PV],
        "cold_target": [BASELINE, FG, PV],
    }
    if suite not in presets:
        raise ValueError(f"Unknown suite {suite}; choose from {list(presets)}")
    return presets[suite]


SUITES = (
    "methods",
    "fg_ablation",
    "fg_count",
    "random_control",
    "pv_ablation",
    "sensitivity",
    "backbone",
    "cold_target",
)
