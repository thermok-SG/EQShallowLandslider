#!/usr/bin/env python3
"""Expand and run a reproducible Cartesian parameter ensemble.

An ordinary model YAML can contain an optional top-level ``ensemble`` block
that maps existing dotted configuration paths to lists of values. Every
Cartesian combination is validated, written as a standalone YAML, and run
through ``run_landslide_model_cli.py``. A digest stored in each run manifest
makes the launcher restartable: completed configurations are skipped unless
``--force`` is supplied.

Use ``--dry-run`` first to validate paths, inspect generated member files, and
confirm the member count without loading a DEM. Random seeds are deliberately
fixed across an ensemble so parameter effects share one stochastic realization.
Concurrent jobs are separate model processes and multiply peak memory use.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import yaml

from run_landslide_model_cli import prepare_config


@dataclass(frozen=True)
class EnsembleMember:
    """One validated model configuration in an ensemble."""

    index: int
    member_id: str
    config_digest: str
    parameters: dict
    config: dict
    config_path: Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Example:
  python run_landslide_ensemble.py --config Synthetic_stability_config.yaml --dry-run
  python run_landslide_ensemble.py --config Synthetic_stability_config.yaml --jobs 2

The YAML 'jobs' value is used unless --jobs is supplied. Use --force only when
you intentionally want to repeat configurations already recorded as complete.""",
    )
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        "--config", help="Path to a model YAML containing an ensemble block."
    )
    config_group.add_argument(
        "--ensemble",
        dest="legacy_ensemble",
        help="Deprecated alias for --config (legacy specification files also work).",
    )
    parser.add_argument(
        "--jobs", type=int, help="Number of model processes to run concurrently."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run all members even when matching completed manifests exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and write member configs without running the model.",
    )
    parser.add_argument(
        "--verbose-console",
        action="store_true",
        help="Forward model log messages to the console as well as member log files.",
    )
    return parser.parse_args(argv)


def _load_mapping(path, description):
    with open(path, encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a YAML mapping at its root")
    return value


def _safe_name(value):
    value = str(value)
    safe = "".join(char if char.isalnum() or char in "-_" else "-" for char in value)
    safe = safe.strip("-")
    if not safe:
        raise ValueError("Ensemble name must contain at least one letter or number")
    return safe


def _set_existing_path(config, dotted_path, value):
    parts = dotted_path.split(".")
    if not all(parts):
        raise ValueError(f"Invalid parameter path: {dotted_path!r}")
    current = config
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            raise ValueError(
                f"Unknown parameter path {dotted_path!r}; {part!r} is not a mapping"
            )
        current = current[part]
    if parts[-1] not in current:
        raise ValueError(f"Unknown parameter path: {dotted_path!r}")
    current[parts[-1]] = value


def _config_digest(config):
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _split_combined_config(config, config_path):
    """Return ensemble settings and the model portion of one combined YAML."""
    if "ensemble" not in config:
        # Keep old two-file specifications usable during migration.
        if "base_config" in config:
            base_value = config.get("base_config")
            if not base_value:
                raise ValueError("base_config is required")
            base_path = Path(base_value)
            if not base_path.is_absolute():
                base_path = config_path.parent / base_path
            return copy.deepcopy(config), _load_mapping(
                base_path.resolve(), "Base configuration"
            )
        raise ValueError(
            "The model configuration must contain a top-level ensemble mapping"
        )

    base = copy.deepcopy(config)
    spec = base.pop("ensemble")
    if not isinstance(spec, dict):
        raise ValueError("ensemble must be a YAML mapping")

    enabled = spec.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("ensemble.enabled must be true or false")
    if not enabled:
        raise ValueError("ensemble.enabled is false; enable it before launching")

    allowed = {
        "enabled",
        "name",
        "jobs",
        "output_dir",
        "log_dir",
        "parameters",
    }
    unknown = set(spec) - allowed
    if unknown:
        raise ValueError(f"Unknown ensemble option(s): {sorted(unknown)}")
    return copy.deepcopy(spec), base


def build_members(spec, spec_path):
    """Expand a Cartesian parameter grid into validated member configurations."""
    spec_path = Path(spec_path).resolve()
    spec, base = _split_combined_config(spec, spec_path)
    name = _safe_name(spec.get("name", spec_path.stem))

    seed = spec.get("random_seed", base.get("random_seed", 5000))
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("random_seed must be one fixed integer")

    grid = spec.get("parameters", {})
    if not isinstance(grid, dict):
        raise ValueError("parameters must be a mapping of dotted paths to lists")
    seed_paths = {"random_seed", "pga.seed"}
    swept_seeds = seed_paths.intersection(grid)
    if swept_seeds:
        raise ValueError(
            "Random seeds are fixed for an ensemble; remove these paths from "
            f"parameters: {sorted(swept_seeds)}"
        )
    for path, values in grid.items():
        if not isinstance(path, str):
            raise ValueError("Every parameter path must be a string")
        if not isinstance(values, list) or not values:
            raise ValueError(f"Parameter {path!r} must contain a non-empty list")

    output_dir = Path(spec.get("output_dir", base.get("output_dir", "./output")))
    output_dir.mkdir(parents=True, exist_ok=True)
    custom_log_dir = spec.get("log_dir")
    if custom_log_dir is not None:
        custom_log_dir = Path(custom_log_dir)
        custom_log_dir.mkdir(parents=True, exist_ok=True)

    paths = list(grid)
    combinations = itertools.product(*(grid[path] for path in paths))
    members = []
    for index, values in enumerate(combinations):
        parameters = dict(zip(paths, values))
        config = copy.deepcopy(base)
        config["random_seed"] = seed
        # Validate against the experiment root first. The member-specific path
        # is operational bookkeeping and is deliberately excluded from the
        # digest so moving an experiment does not change its scientific identity.
        config["output_dir"] = str(output_dir)
        for path, value in parameters.items():
            _set_existing_path(config, path, value)

        config = prepare_config(config)
        digest_config = copy.deepcopy(config)
        digest_config.pop("output_dir", None)
        digest_config.pop("log_file", None)
        digest = _config_digest(digest_config)
        member_id = f"member-{index:04d}-{digest[:10]}"
        member_dir = output_dir / "members" / member_id
        member_dir.mkdir(parents=True, exist_ok=True)
        config["output_dir"] = str(member_dir)
        config["ensemble"] = {
            "name": name,
            "member_id": member_id,
            "member_index": index,
            "config_digest": digest,
            "parameters": parameters,
        }
        config["log_file"] = str(
            custom_log_dir / f"{member_id}.log"
            if custom_log_dir is not None
            else member_dir / "run.log"
        )
        config_path = member_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
        members.append(
            EnsembleMember(
                index=index,
                member_id=member_id,
                config_digest=digest,
                parameters=parameters,
                config=config,
                config_path=config_path.resolve(),
            )
        )
    ensemble_manifest = {
        "schema_version": 1,
        "name": name,
        "source_config": str(spec_path),
        "random_seed": seed,
        "parameters": grid,
        "member_count": len(members),
        "members": [
            {
                "index": member.index,
                "member_id": member.member_id,
                "config_digest": member.config_digest,
                "parameters": member.parameters,
                "config": str(member.config_path.relative_to(output_dir.resolve())),
            }
            for member in members
        ],
    }
    with open(output_dir / "ensemble_manifest.json", "w", encoding="utf-8") as stream:
        json.dump(ensemble_manifest, stream, indent=2)
    return members


def completed_digests(output_dir, ensemble_name):
    """Find successfully saved members of this ensemble from their manifests."""
    completed = set()
    for manifest_path in Path(output_dir).rglob("manifest.json"):
        try:
            with open(manifest_path, encoding="utf-8") as stream:
                manifest = json.load(stream)
            metadata = manifest.get("config", {}).get("ensemble", {})
            if metadata.get("name") == ensemble_name and metadata.get("config_digest"):
                completed.add(metadata["config_digest"])
        except (OSError, json.JSONDecodeError, TypeError):
            continue
    return completed


def run_member(member, verbose_console=False):
    command = [
        sys.executable,
        str(Path(__file__).with_name("run_landslide_model_cli.py")),
        "--config",
        str(member.config_path),
    ]
    if verbose_console:
        command.append("--verbose_console")
    result = subprocess.run(command, check=False)
    return result.returncode


def run_members(members, jobs=1, verbose_console=False):
    """Run members concurrently and return ``(member, returncode)`` pairs."""
    if jobs < 1:
        raise ValueError("jobs must be at least 1")
    results = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {}
        for member in members:
            print(f"[{member.index + 1}] starting {member.member_id}", flush=True)
            future = executor.submit(run_member, member, verbose_console)
            futures[future] = member
        for future in as_completed(futures):
            member = futures[future]
            returncode = future.result()
            status = "completed" if returncode == 0 else f"failed (exit {returncode})"
            print(f"[{member.index + 1}] {status}: {member.member_id}", flush=True)
            results.append((member, returncode))
    return results


def main(argv=None):
    args = parse_args(argv)
    config_value = args.config or args.legacy_ensemble
    spec_path = Path(config_value).resolve()
    combined_config = _load_mapping(spec_path, "Model configuration")
    spec, base_config = _split_combined_config(combined_config, spec_path)
    members = build_members(combined_config, spec_path)
    name = _safe_name(spec.get("name", spec_path.stem))
    if members:
        output_dir = Path(
            spec.get("output_dir", base_config.get("output_dir", "./output"))
        )
        fixed_seed = spec.get("random_seed", members[0].config["random_seed"])
    else:
        output_dir = Path("./output")
        fixed_seed = spec.get("random_seed", 5000)
    jobs = args.jobs if args.jobs is not None else spec.get("jobs", 1)
    if not isinstance(jobs, int) or isinstance(jobs, bool) or jobs < 1:
        raise ValueError("jobs must be a positive integer")

    completed = set() if args.force else completed_digests(output_dir, name)
    pending = [member for member in members if member.config_digest not in completed]
    print(
        f"Ensemble {name}: {len(members)} members, "
        f"{len(members) - len(pending)} completed, {len(pending)} pending; "
        f"fixed seed={fixed_seed}",
        flush=True,
    )
    if args.dry_run:
        print("Dry run complete; member configurations were written.", flush=True)
        return 0
    if not pending:
        print("Nothing to run.", flush=True)
        return 0

    results = run_members(pending, jobs=jobs, verbose_console=args.verbose_console)
    failures = [member for member, returncode in results if returncode != 0]
    if failures:
        print(f"{len(failures)} ensemble member(s) failed.", file=sys.stderr)
        return 1
    print(f"All {len(results)} pending ensemble members completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
