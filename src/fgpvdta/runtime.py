"""Explicit, interpreter-bound setup profiles; not licensing or access control."""

import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

SCHEMA = "fgpvdta.runtime.v1"
CORE_REQUIREMENTS = {
    "numpy": "1.26.4",
    "rdkit": "2024.3.6",
    "torch": "2.4.1",
    "torch-geometric": "2.6.1",
}
PV_REQUIREMENTS = {"torchvision": "0.19.1", "pillow": "10.4.0"}


def inspect_runtime(methods):
    methods = sorted(set(methods))
    if not methods or set(methods) - {"fg", "pv"}:
        raise ValueError("Select one or both method families: fg, pv")
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError(
            "Research execution requires Python 3.10; create a dedicated environment"
        )
    requirements = dict(CORE_REQUIREMENTS)
    if "pv" in methods:
        requirements.update(PV_REQUIREMENTS)
    installed, errors = {}, []
    for name, expected in requirements.items():
        try:
            actual = version(name)
        except PackageNotFoundError:
            actual = "missing"
        # PyTorch CUDA/CPU wheel tags do not change the pinned release version.
        if actual.split("+", 1)[0] != expected:
            errors.append(f"{name}=={expected} required; found {actual}")
        installed[name] = actual
    if errors:
        raise RuntimeError("Runtime requirements not satisfied:\n" + "\n".join(errors))
    return {
        "schema": SCHEMA,
        "methods": methods,
        "python": platform.python_version(),
        "executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "packages": installed,
    }


def configure_runtime(output, methods):
    payload = inspect_runtime(methods)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Profiles must be deliberately recreated; never silently overwrite one.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return path


def require_runtime(profile, methods):
    if not profile:
        raise RuntimeError(
            "Manual runtime setup is required. Run `fgpvdta configure-runtime "
            "--methods fg pv --output .runtime/research.json`, then pass "
            "--runtime-profile .runtime/research.json to train, suite or predict."
        )
    path = Path(profile)
    if not path.is_file():
        raise FileNotFoundError(f"Runtime profile not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Invalid runtime profile JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("Unsupported runtime profile; create one with configure-runtime")
    enabled = payload.get("methods")
    if not isinstance(enabled, list) or any(m not in ("fg", "pv") for m in enabled):
        raise ValueError("Invalid runtime method families")
    if not set(methods).issubset(enabled):
        raise RuntimeError("Runtime profile does not enable all requested method families")
    current = inspect_runtime(enabled)
    if payload != current:
        raise RuntimeError(
            "Runtime changed or profile belongs to another environment. "
            "Create a new profile explicitly using configure-runtime."
        )
    return current
