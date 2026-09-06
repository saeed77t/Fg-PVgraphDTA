import json
import sys
from importlib.metadata import PackageNotFoundError

import pytest

from fgpvdta import runtime
from fgpvdta.cli import main


@pytest.fixture
def pinned(monkeypatch):
    versions = {**runtime.CORE_REQUIREMENTS, **runtime.PV_REQUIREMENTS}
    monkeypatch.setattr(runtime, "version", versions.__getitem__)
    monkeypatch.setattr(sys, "version_info", (3, 10, 19))
    return versions


def test_missing_profile_stops_before_loading_data():
    with pytest.raises(RuntimeError, match="Manual runtime setup"):
        main(["train", "--model", "fggraphdta", "--data", "does-not-exist"])


def test_profile_roundtrip_and_no_overwrite(tmp_path, pinned):
    profile = tmp_path / "runtime.json"
    runtime.configure_runtime(profile, ["fg", "pv"])
    assert runtime.require_runtime(profile, ["fg", "pv"])["schema"] == runtime.SCHEMA
    with pytest.raises(FileExistsError):
        runtime.configure_runtime(profile, ["fg"])


def test_fg_profile_does_not_enable_pv(tmp_path, pinned):
    profile = runtime.configure_runtime(tmp_path / "runtime.json", ["fg"])
    with pytest.raises(RuntimeError, match="does not enable"):
        runtime.require_runtime(profile, ["pv"])


def test_wrong_dependency_rejected(pinned):
    pinned["torch"] = "9.0.0"
    with pytest.raises(RuntimeError, match="torch==2.4.1"):
        runtime.inspect_runtime(["fg"])


def test_missing_dependency_rejected(pinned, monkeypatch):
    def missing(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(runtime, "version", missing)
    with pytest.raises(RuntimeError, match="found missing"):
        runtime.inspect_runtime(["fg"])


def test_other_python_rejected(pinned, monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 11, 0))
    with pytest.raises(RuntimeError, match="Python 3.10"):
        runtime.inspect_runtime(["fg"])


def test_changed_interpreter_rejected(tmp_path, pinned):
    profile = runtime.configure_runtime(tmp_path / "runtime.json", ["fg"])
    payload = json.loads(profile.read_text())
    payload["executable"] = "another/python"
    profile.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="another environment"):
        runtime.require_runtime(profile, ["fg"])


def test_torch_build_tag_accepted(pinned):
    pinned["torch"] = "2.4.1+cu121"
    assert runtime.inspect_runtime(["fg"])["packages"]["torch"] == "2.4.1+cu121"


def test_invalid_profile_rejected(tmp_path, pinned):
    profile = tmp_path / "runtime.json"
    profile.write_text("[]")
    with pytest.raises(ValueError, match="Unsupported runtime profile"):
        runtime.require_runtime(profile, ["fg"])
