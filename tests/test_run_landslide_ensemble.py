import json
from pathlib import Path

import pytest
import yaml

import run_landslide_ensemble as ensemble


def base_config(tmp_path):
    return {
        "dem_path": "dem.asc",
        "output_dir": str(tmp_path / "base-runs"),
        "random_seed": 11,
        "chunking": {"mode": "auto"},
        "soil_params": {
            "cohesion_eff": 15000,
            "angle_int_frict": 30,
        },
        "pga": {},
        "simulation": {},
        "flow_params": {},
        "split_by_width": {},
        "outputs": {},
    }


def write_config(tmp_path, parameters=None, **overrides):
    config = base_config(tmp_path)
    config["ensemble"] = {
        "enabled": True,
        "name": "test ensemble",
        "output_dir": str(tmp_path / "runs"),
        "parameters": parameters or {},
    }
    config["ensemble"].update(overrides)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path, config


def test_build_members_expands_cartesian_grid_with_fixed_seed(tmp_path):
    spec_path, spec = write_config(
        tmp_path,
        {
            "soil_params.cohesion_eff": [10000, 20000],
            "soil_params.angle_int_frict": [25, 35],
        },
    )

    members = ensemble.build_members(spec, spec_path)

    assert len(members) == 4
    assert {member.config["random_seed"] for member in members} == {11}
    assert {
        (
            member.config["soil_params"]["cohesion_eff"],
            member.config["soil_params"]["angle_int_frict"],
        )
        for member in members
    } == {(10000, 25), (10000, 35), (20000, 25), (20000, 35)}
    assert len({member.config_digest for member in members}) == 4
    assert all(member.config_path.exists() for member in members)
    assert all(member.config_path.name == "config.yaml" for member in members)
    assert all(member.config_path.parent.parent.name == "members" for member in members)
    assert all(
        Path(member.config["output_dir"]) == member.config_path.parent
        for member in members
    )
    assert all(member.config["log_file"].endswith("run.log") for member in members)
    assert (tmp_path / "runs" / "ensemble_manifest.json").exists()


def test_build_members_rejects_seed_sweep_and_unknown_path(tmp_path):
    spec_path, spec = write_config(tmp_path, {"random_seed": [1, 2]})
    with pytest.raises(ValueError, match="fixed"):
        ensemble.build_members(spec, spec_path)

    spec_path, spec = write_config(tmp_path, {"pga.seed": [1, 2]})
    with pytest.raises(ValueError, match="fixed"):
        ensemble.build_members(spec, spec_path)

    spec_path, spec = write_config(tmp_path, {"soil_params.typo": [1]})
    with pytest.raises(ValueError, match="Unknown parameter path"):
        ensemble.build_members(spec, spec_path)


def test_completed_digests_only_uses_matching_ensemble(tmp_path):
    run_root = tmp_path / "runs"
    matching = run_root / "matching"
    matching.mkdir(parents=True)
    (matching / "manifest.json").write_text(
        json.dumps(
            {
                "config": {
                    "ensemble": {
                        "name": "target",
                        "config_digest": "complete-digest",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    other = run_root / "other"
    other.mkdir()
    (other / "manifest.json").write_text(
        json.dumps(
            {"config": {"ensemble": {"name": "other", "config_digest": "ignored"}}}
        ),
        encoding="utf-8",
    )
    broken = run_root / "broken"
    broken.mkdir()
    (broken / "manifest.json").write_text("not json", encoding="utf-8")

    assert ensemble.completed_digests(run_root, "target") == {"complete-digest"}


def test_dry_run_writes_configs_without_starting_models(tmp_path, monkeypatch):
    spec_path, _ = write_config(tmp_path, {"soil_params.cohesion_eff": [10000, 20000]})

    def fail_if_called(*args, **kwargs):
        raise AssertionError("dry run must not start model processes")

    monkeypatch.setattr(ensemble, "run_members", fail_if_called)

    assert ensemble.main(["--config", str(spec_path), "--dry-run"]) == 0
    assert len(list((tmp_path / "runs" / "members").glob("*/config.yaml"))) == 2


def test_model_lists_are_not_mistaken_for_sweep_parameters(tmp_path):
    spec_path, config = write_config(
        tmp_path, {"soil_params.cohesion_eff": [10000, 20000]}
    )
    config["chunking"]["tile_size"] = [100, 200]

    members = ensemble.build_members(config, spec_path)

    assert len(members) == 2
    assert all(
        member.config["chunking"]["tile_size"] == [100, 200] for member in members
    )


def test_combined_config_requires_enabled_ensemble_mapping(tmp_path):
    config = base_config(tmp_path)
    with pytest.raises(ValueError, match="top-level ensemble"):
        ensemble.build_members(config, tmp_path / "config.yaml")

    config["ensemble"] = {"enabled": False, "parameters": {}}
    with pytest.raises(ValueError, match="enabled is false"):
        ensemble.build_members(config, tmp_path / "config.yaml")


def test_run_members_reports_subprocess_failures(monkeypatch, tmp_path):
    members = [
        ensemble.EnsembleMember(
            index=index,
            member_id=f"member-{index}",
            config_digest=str(index),
            parameters={},
            config={},
            config_path=Path(tmp_path / f"{index}.yaml"),
        )
        for index in range(2)
    ]
    monkeypatch.setattr(
        ensemble,
        "run_member",
        lambda member, verbose_console=False: 1 if member.index == 1 else 0,
    )

    results = ensemble.run_members(members, jobs=2)

    assert {member.index: code for member, code in results} == {0: 0, 1: 1}
