import numpy as np
import pytest

from generate_synthetic_topography import (
    generate_mountain_catchment,
    write_esri_ascii,
)


def test_synthetic_catchment_is_reproducible_and_has_mountain_relief():
    first, stats = generate_mountain_catchment(
        40, 40, spacing=30, seed=17, iterations=3, refinement_factor=2
    )
    second, _ = generate_mountain_catchment(
        40, 40, spacing=30, seed=17, iterations=3, refinement_factor=2
    )

    first_z = first.at_node["topographic__elevation"]
    second_z = second.at_node["topographic__elevation"]
    assert np.array_equal(first_z, second_z)
    assert first.shape == (40, 40)
    assert np.isfinite(first_z).all()
    assert stats["cell_count"] == 1600
    assert stats["source_evolution_shape"] == [20, 20]
    assert stats["generator"] == "Landlab tutorial-based mountain-range evolution"
    assert stats["components"] == [
        "PriorityFloodFlowRouter (D8)",
        "SpaceLargeScaleEroder",
    ]
    assert stats["elevation_relief_m"] > 0
    assert stats["maximum_drainage_area_m2"] > 30**2


def test_synthetic_seed_changes_roughness():
    first, _ = generate_mountain_catchment(
        40, 40, seed=1, iterations=1, refinement_factor=2
    )
    second, _ = generate_mountain_catchment(
        40, 40, seed=2, iterations=1, refinement_factor=2
    )
    assert not np.array_equal(
        first.at_node["topographic__elevation"],
        second.at_node["topographic__elevation"],
    )


def test_write_esri_ascii_header_and_values(tmp_path):
    grid, _ = generate_mountain_catchment(
        40, 40, iterations=1, refinement_factor=2
    )
    output = tmp_path / "terrain.asc"
    write_esri_ascii(output, grid)

    lines = output.read_text(encoding="utf-8").splitlines()
    assert set(lines[:2]) == {"NROWS 40", "NCOLS 40"}
    assert "CELLSIZE 30.0" in lines[:6]
    values = np.loadtxt(output, skiprows=6)
    assert values.shape == grid.shape
    assert np.count_nonzero(values == -9999) == 0


@pytest.mark.parametrize("shape", [(19, 100), (100, 19)])
def test_synthetic_catchment_rejects_tiny_grids(shape):
    with pytest.raises(ValueError, match="at least 20"):
        generate_mountain_catchment(*shape)
