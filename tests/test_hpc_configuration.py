import subprocess
from pathlib import Path

import pytest
import yaml

REPOSITORY_ROOT = Path(__file__).parents[1]
HPC_ROOT = REPOSITORY_ROOT / "hpc"


@pytest.mark.parametrize("region", ["nepal", "japan"])
def test_hpc_config_uses_paths_relative_to_workspace(region):
    with open(
        HPC_ROOT / "configs" / f"{region}_config.yaml", encoding="utf-8"
    ) as stream:
        config = yaml.safe_load(stream)

    input_paths = [
        config["dem_path"],
        config["split_by_width"]["pickle_path"],
        *config["split_by_width"]["csv_paths"].values(),
    ]
    output_paths = [config["output_dir"], config["ensemble"]["output_dir"]]
    assert all(path.startswith("input_data/") for path in input_paths)
    assert all(path.startswith("runs/") for path in output_paths)


def test_hpc_shell_scripts_are_valid_and_portable():
    scripts = [
        HPC_ROOT / "setup_workspace.sh",
        *sorted((HPC_ROOT / "slurm").glob("*.sbatch")),
    ]
    assert scripts

    for script in scripts:
        subprocess.run(["bash", "-n", script], check=True)
        contents = script.read_text(encoding="utf-8")
        assert "EQShallowLandslider_HPC" not in contents
        assert "/users/" not in contents
