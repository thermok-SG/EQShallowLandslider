
import numpy as np
import pytest
from landlab import RasterModelGrid

from helper_functions import (
    # regions
    calculate_regions,
    split_groups_by_aspect,
    _create_zones,
    calculate_region_properties,
    # selection
    generate_landslide_probability,
    probabilistic_group_selection,
    generate_landslide_proportion_from_pga,
    select_groups_by_proportion_weighted,
    # split
    recursive_split_wide_regions,
)


def small_grid(shape=(6, 6), spacing=10.0):
    mg = RasterModelGrid(shape, xy_spacing=spacing)
    z = mg.add_zeros("topographic__elevation", at="node")
    z[:] = np.arange(z.size)
    return mg


# --- calculate_regions ---
def test_calculate_regions_connectivity_4_vs_8():
    mg = small_grid()
    mask = np.zeros(mg.shape, dtype=bool)
    mask[1, 1] = True
    mask[2, 2] = True  # diagonal neighbor
    labels4, n4 = calculate_regions(mask, connect_val=4)
    labels8, n8 = calculate_regions(mask, connect_val=8)
    assert n4 == 2
    assert n8 == 1


def test_calculate_regions_weight_errors():
    mg = small_grid()
    mask = np.zeros(mg.shape, dtype=bool)
    mask[2:4, 2:4] = True
    with pytest.raises(ValueError):
        calculate_regions(
            mask,
            proximity_weight_on=True,
            density_weight_on=True,
            proximity_weight_val=0.6,
            density_weight_val=0.5,
        )


# --- split_groups_by_aspect ---
def test_split_groups_by_aspect_basic():
    mg = small_grid()
    # one big group
    groups = np.zeros(mg.shape, dtype=int)
    groups[1:5, 1:5] = 1
    # aspect: north vs south halves
    aspect = np.zeros(mg.shape, dtype=float)
    aspect[:3, :] = 10.0   # near north
    aspect[3:, :] = 190.0  # near south
    zones = _create_zones(90)
    new_groups, zone_labels, info = split_groups_by_aspect(
        groups, aspect, zones=zones, min_size=2
    )
    assert new_groups.max() >= 2
    assert np.any(new_groups[:3, :] > 0) and np.any(new_groups[3:, :] > 0)


# --- calculate_region_properties ---
def test_calculate_region_properties_returns_df_and_labels():
    mg = small_grid()
    groups = np.zeros(mg.shape, dtype=int)
    groups[1:3, 1:3] = 1

    slopes = np.ones(mg.number_of_nodes) * 10.0          # 1-D is fine; the function reshapes this
    aspect = np.ones(mg.shape) * 45.0                    # <<< 2-D aspect matching mg.shape

    df, working = calculate_region_properties(mg, groups, slopes, aspect)
    assert not df.empty
    assert "area" in df.columns and "orientation" in df.columns



# --- generate_landslide_probability ---

def test_generate_landslide_probability_normalize_flag():
    mg = small_grid()
    labeled = np.zeros(mg.shape, dtype=int)
    labeled[1:3, 1:3] = 1
    labeled[3:5, 3:5] = 2

    # Base arrays
    h = np.ones(mg.number_of_nodes) * 0.30  # baseline horizontal PGA (g)
    v = np.ones(mg.number_of_nodes) * 0.10  # baseline vertical PGA (g)
    slope = np.ones(mg.number_of_nodes) * 20.0
    crit = np.ones(mg.number_of_nodes) * 0.20

    # Make group 2 "stronger shaking": increase both h and v for label==2
    mask2 = (labeled.ravel() == 2)
    h[mask2] = 0.60
    v[mask2] = 0.15

    # Raw (no normalization)
    probs_raw, meta_raw = generate_landslide_probability(
        mg, h, v, labeled, slope_array=slope, critical_acceleration_array=crit, normalise_final_probs=False
    )

    # Normalized across groups
    probs_norm, meta_norm = generate_landslide_probability(
        mg, h, v, labeled, slope_array=slope, critical_acceleration_array=crit, normalise_final_probs=True
    )

    # Assert normalization was performed and arrays differ
    assert meta_norm["normalization"].get("performed", False)
    assert np.any(probs_raw != probs_norm)



# --- probabilistic_group_selection ---
def test_probabilistic_group_selection_reproducible_seed():
    mg = small_grid()
    labeled = np.zeros(mg.shape, dtype=int)
    labeled[1:3, 1:3] = 1
    labeled[3:5, 3:5] = 2
    probs = np.zeros_like(labeled, dtype=float)
    probs[labeled == 1] = 0.8
    probs[labeled == 2] = 0.2

    sel1, meta1 = probabilistic_group_selection(
        labeled, probs, random_seed=123, reproducible=True
    )
    sel2, meta2 = probabilistic_group_selection(
        labeled, probs, random_seed=123, reproducible=True
    )
    assert np.array_equal(sel1, sel2)
    assert meta1["proportion_calculated"] == meta2["proportion_calculated"]


# --- generate_landslide_proportion_from_pga ---
def test_generate_landslide_proportion_from_pga_shapes():
    mg = small_grid()
    labeled = np.zeros(mg.shape, dtype=int)
    labeled[1:3, 1:3] = 1
    h = np.ones(mg.number_of_nodes) * 0.2
    v = np.ones(mg.number_of_nodes) * 0.1
    probs, prop, meta = generate_landslide_proportion_from_pga(
        mg, h, v, labeled, weight_array=np.ones_like(h)
    )
    assert probs.shape == labeled.shape
    assert 0.0 <= prop <= 1.0


# --- recursive_split_wide_regions ---
class FakeKDE:
    def resample(self, n):
        # Return 2xN array: [length, width] in transformed space (no log)
        # Simulate widths around 10, with lengths around 5
        return np.vstack([np.ones(n) * 5.0, np.ones(n) * 10.0])


def test_recursive_split_wide_regions_triggers_split_when_ratio_exceeds_threshold():
    mg = small_grid()
    labeled = np.zeros(mg.shape, dtype=int)
    labeled[1:5, 1:5] = 1
    aspect = np.ones(mg.shape) * 45
    slopes = np.ones(mg.shape) * 20
    kde_results = {"overall": FakeKDE()}
    transform_info = {"log_x": False, "log_y": False}

    # Force width to appear much larger than expected (threshold very low)
    new_labels, info = recursive_split_wide_regions(
        grid=mg,
        labeled_array=labeled,
        aspect_array=aspect,
        slopes_grid=slopes,
        kde_results=kde_results,
        transform_info=transform_info,
        width_threshold=0.5,
        max_iterations=1,
        min_region_size=5,
        convergence_threshold=0.95,
    )
    assert new_labels.max() > 1