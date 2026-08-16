import json
import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from landlab import RasterModelGrid

from analysis import (
    compare_run_distributions,
    discover_runs,
    load_observed_landslides,
    load_region_ensemble,
    load_run,
    plot_run,
    plot_run_maps,
    plot_parameter_sensitivity,
    summarize_run_distributions,
    swept_parameters,
)
from utils.utilities import save_model_run


def make_completed_run():
    grid = RasterModelGrid((4, 4), xy_spacing=30.0)
    grid.add_field("topographic__elevation", np.arange(16, dtype=float), at="node")
    grid.add_ones("soil__depth", at="node")
    labels = np.zeros(16, dtype=int)
    labels[[5, 6]] = 1
    labels[[9, 10, 13]] = 2
    selected = np.where(labels == 2, labels, 0)
    properties = pd.DataFrame(
        {
            "area": [1800.0, 2700.0],
            "median_slope": [20.0, 30.0],
            "median_elevation": [5.5, 10.0],
            "max_elevation": [6.0, 13.0],
            "local_relief": [1.0, 4.0],
            "mean_aspect": [90.0, 180.0],
            "slope_direction_length_new": [40.0, 60.0],
            "perpendicular_width_new": [25.0, 35.0],
        },
        index=pd.Index([1, 2], name="label"),
    )
    results = {
        "group_properties": properties,
        "factor_of_safety": np.linspace(0.5, 2.0, 16),
        "a_transient": np.linspace(1.0, 2.0, 16),
        "a_driving": np.linspace(0.5, 2.5, 16),
        "a_diff": np.linspace(-0.5, 0.5, 16),
        "unstable_mask": labels > 0,
        "labels": labels,
        "aspect_labels": labels,
        "split_labels": labels,
        "selected_labels": selected,
        "selected_proportion": 0.5,
        "newmark": np.where(selected > 0, 0.25, np.nan),
        "runout": None,
    }
    return SimpleNamespace(grid=grid, results=results)


def make_config():
    return {
        "dem_path": "missing-test-dem.asc",
        "output_dir": "unused",
        "random_seed": 42,
        "soil_params": {
            "cohesion_eff": 15000,
            "angle_int_frict": 30,
            "submerged_soil_proportion": 0.5,
            "distribution": "curvature",
            "relationship": "linear_std_local",
        },
        "simulation": {"selection_method": "probabilistic"},
        "outputs": {
            "write_parquet": False,
            "write_zarr": False,
            "write_npy_fallback": True,
        },
    }


def test_save_model_run_writes_v12_analysis_bundle(tmp_path):
    run_dir = save_model_run(
        save_pickle=False,
        ls=make_completed_run(),
        config=make_config(),
        output_dir=tmp_path,
        logger=logging.getLogger("output-test"),
        runtime_metadata={"execution_mode": "chunked"},
    )

    regions = pd.read_csv(run_dir / "regions.csv")
    assert regions["label"].tolist() == [1, 2]
    assert regions["selected"].tolist() == [False, True]
    assert regions["cell_count"].tolist() == [2, 3]
    assert np.isclose(regions.loc[1, "max_newmark_displacement"], 0.25)

    with open(run_dir / "manifest.json", encoding="utf-8") as stream:
        manifest = json.load(stream)
    assert manifest["model"]["version"] == "1.2.0"
    assert manifest["runtime"]["execution_mode"] == "chunked"
    assert manifest["grid"]["shape"] == [4, 4]
    assert (run_dir / "rasters" / "selected_labels.npy").exists()

    with open(run_dir / "summary.json", encoding="utf-8") as stream:
        summary = json.load(stream)
    assert summary["candidate_region_count"] == 2
    assert summary["selected_region_count"] == 1
    assert summary["selected_area_m2"] == 2700.0
    assert summary["selected_footprint_node_count"] == 3
    assert summary["selected_footprint_area_m2"] == 2700.0
    assert summary["affected_node_percent"] == summary["selected_footprint_percent"]
    assert summary["runout_enabled"] is False


def test_save_model_run_includes_runout_diagnostics(tmp_path):
    completed = make_completed_run()
    completed.results["runout"] = {
        "failed_nodes": np.array([9, 10, 13]),
        "paths": [(9, 5), (9, 6), (10, 6), (13, 9)],
        "source_proportion_sums": {9: 1.0, 10: 1.0, 13: 1.0},
        "source_path_counts": {9: 2, 10: 1, 13: 1},
    }
    completed.grid.add_field(
        "landslide__erosion", np.linspace(0.0, 0.15, 16), at="node"
    )
    completed.grid.add_field(
        "landslide__deposition", np.linspace(0.15, 0.0, 16), at="node"
    )
    completed.grid.add_field(
        "landslide__soil_depth_change",
        completed.grid.at_node["landslide__deposition"]
        - completed.grid.at_node["landslide__erosion"],
        at="node",
    )

    run_dir = save_model_run(
        False,
        completed,
        make_config(),
        tmp_path,
        logging.getLogger("runout-output-test"),
    )
    loaded = load_run(run_dir, load_rasters=True)

    assert "runout_erosion" in loaded["rasters"]
    assert "runout_deposition" in loaded["rasters"]
    assert "runout_soil_depth_change" in loaded["rasters"]
    assert "selected_footprint" in loaded["rasters"]
    assert "runout_affected_footprint" in loaded["rasters"]
    assert "runout_only_footprint" in loaded["rasters"]
    assert "combined_affected_footprint" in loaded["rasters"]
    assert loaded["summary"]["runout_enabled"] is True
    assert loaded["summary"]["runout_changed_node_count"] == 16
    assert loaded["summary"]["runout_source_node_count"] == 3
    assert loaded["summary"]["runout_moving_source_node_count"] == 3
    assert loaded["summary"]["runout_terminated_path_count"] == 4
    assert np.isclose(loaded["summary"]["runout_mean_paths_per_moving_source"], 4 / 3)
    assert loaded["summary"]["runout_max_paths_per_source"] == 2
    assert loaded["summary"]["runout_source_proportion_error_count"] == 0
    assert loaded["summary"]["runout_affected_footprint_node_count"] == 16
    assert loaded["summary"]["runout_only_footprint_node_count"] == 13
    assert loaded["summary"]["selected_and_runout_overlap_node_count"] == 3
    assert loaded["summary"]["combined_affected_footprint_node_count"] == 16
    assert np.isclose(loaded["summary"]["runout_mass_balance_error_node_m"], 0.0)
    assert loaded["summary"]["negative_final_soil_depth_node_count"] == 0


def test_analysis_loaders_and_plot(tmp_path):
    member_dir = tmp_path / "members" / "member-0000-example"
    member_dir.mkdir(parents=True)
    run_dir = save_model_run(
        False,
        make_completed_run(),
        make_config(),
        member_dir,
        logging.getLogger("analysis-test"),
    )

    assert discover_runs(tmp_path) == [run_dir]
    loaded = load_run(run_dir, load_rasters=True)
    assert loaded["rasters"]["selected_labels"].shape == (4, 4)
    ensemble = load_region_ensemble(tmp_path, selected_only=True)
    assert ensemble["label"].tolist() == [2]

    figure = plot_run(loaded, output_path=tmp_path / "plot.png")
    assert (tmp_path / "plot.png").exists()
    plt.close(figure)


def test_parameter_sensitivity_holds_other_swept_parameters_fixed(tmp_path):
    for index, (cohesion, friction) in enumerate(
        ((10000, 25), (20000, 25), (10000, 35), (20000, 35))
    ):
        config = make_config()
        config["soil_params"].update(
            {"cohesion_eff": cohesion, "angle_int_frict": friction}
        )
        config["ensemble"] = {
            "name": "controlled-test",
            "member_id": f"member-{index:04d}",
            "config_digest": f"digest-{index}",
            "parameters": {
                "soil_params.cohesion_eff": cohesion,
                "soil_params.angle_int_frict": friction,
            },
        }
        save_model_run(
            False,
            make_completed_run(),
            config,
            tmp_path / "members" / f"member-{index:04d}",
            logging.getLogger("parameter-sensitivity-test"),
        )

    plot_dir = tmp_path / "analysis" / "cohesion"
    assert swept_parameters(tmp_path) == [
        "soil_params.angle_int_frict",
        "soil_params.cohesion_eff",
    ]
    table = plot_parameter_sensitivity(
        tmp_path, "cohesion_eff", plot_dir, selected_only=True
    )

    assert sorted(plot_dir.glob("*.png")) == [
        plot_dir / "soil_params-cohesion_eff_001.png",
        plot_dir / "soil_params-cohesion_eff_002.png",
    ]
    assert table["comparison_id"].nunique() == 2
    assert set(table["parameter"]) == {"soil_params.cohesion_eff"}
    assert set(table["parameter_value"]) == {10000, 20000}
    assert set(table["fixed_parameters"]) == {
        '{"soil_params.angle_int_frict":25}',
        '{"soil_params.angle_int_frict":35}',
    }
    assert set(table["count"]) == {1}


def test_parameter_sensitivity_overlays_observations_with_readable_titles(
    tmp_path, monkeypatch
):
    for index, cohesion in enumerate((10000, 20000)):
        config = make_config()
        config["soil_params"]["cohesion_eff"] = cohesion
        config["ensemble"] = {
            "name": "readable_title_test",
            "member_id": f"member-{index:04d}",
            "config_digest": f"digest-{index}",
            "parameters": {
                "soil_params.cohesion_eff": cohesion,
                "soil_params.angle_int_frict": 30,
            },
        }
        save_model_run(
            False,
            make_completed_run(),
            config,
            tmp_path / "members" / f"member-{index:04d}",
            logging.getLogger("observed-sensitivity-test"),
        )

    observed = pd.DataFrame(
        {
            "area": [1200.0, 2400.0],
            "median_slope": [22.0, 28.0],
        }
    )
    table = plot_parameter_sensitivity(
        tmp_path,
        "cohesion_eff",
        tmp_path / "analysis",
        observed=observed,
    )

    assert (tmp_path / "analysis" / "soil_params-cohesion_eff_001.png").exists()
    # Exercise the public plot with a retained figure by intercepting close.
    with monkeypatch.context() as patch:
        patch.setattr(plt, "close", lambda _figure: None)
        plot_parameter_sensitivity(
            tmp_path,
            "cohesion_eff",
            tmp_path / "analysis-retained",
            observed=observed,
        )
        plotted = plt.gcf()
        assert "Sensitivity to Effective cohesion" in plotted._suptitle.get_text()
        labels = [line.get_label() for line in plotted.axes[0].lines]
        assert "Measured (n=2)" in labels
        measured = next(
            line
            for line in plotted.axes[0].lines
            if line.get_label() == "Measured (n=2)"
        )
        assert measured.get_linestyle() == "--"
    plt.close(plotted)


def test_observed_landslides_are_normalized_filtered_and_plotted(tmp_path):
    inventory_path = tmp_path / "inventory.csv"
    pd.DataFrame(
        {
            "area_m2": [500.0, 1800.0, 2700.0],
            "length_m": [20.0, 50.0, 70.0],
            "width_m": [10.0, 30.0, 40.0],
        }
    ).to_csv(inventory_path, index=False)
    zonal_path = tmp_path / "zonal.csv"
    pd.DataFrame(
        {
            "Area_m2": [500.0, 1800.0, 2700.0],
            "Elevation_m_mean": [100.0, 200.0, 300.0],
            "Slope_deg_mean": [15.0, 25.0, 35.0],
        }
    ).to_csv(zonal_path, index=False)

    observed = load_observed_landslides(inventory_path, zonal_path, min_area=1000.0)
    assert observed["area"].tolist() == [1800.0, 2700.0]
    assert observed["median_slope"].tolist() == [25.0, 35.0]
    assert observed["slope_direction_length_new"].tolist() == [50.0, 70.0]

    run_dir = save_model_run(
        False,
        make_completed_run(),
        make_config(),
        tmp_path / "runs",
        logging.getLogger("observed-analysis-test"),
    )
    run = load_run(run_dir)
    figure = plot_run(run, observed=observed, output_path=tmp_path / "comparison.png")
    assert (tmp_path / "comparison.png").exists()
    summary = summarize_run_distributions(run, observed=observed)
    area = summary[summary["metric"] == "area"].set_index("source")
    assert area.loc["model", "count"] == 1
    assert area.loc["observed", "count"] == 2
    comparison = compare_run_distributions(run, observed)
    area_comparison = comparison.set_index("metric").loc["area"]
    assert 0 <= area_comparison["ks_statistic"] <= 1
    assert 0 <= area_comparison["kuiper_statistic"] <= 2
    assert area_comparison["wasserstein_distance"] >= 0

    map_figure = plot_run_maps(run, output_path=tmp_path / "maps.png")
    assert (tmp_path / "maps.png").exists()
    plt.close(map_figure)
    plt.close(figure)


def test_optional_storage_formats_have_safe_fallbacks(tmp_path):
    config = make_config()
    config["outputs"].update({"write_parquet": True, "write_zarr": True})

    run_dir = save_model_run(
        False,
        make_completed_run(),
        config,
        tmp_path,
        logging.getLogger("storage-fallback-test"),
    )

    assert (run_dir / "regions.csv").exists()
    assert (run_dir / "rasters.zarr").exists() or (
        run_dir / "rasters" / "metadata.json"
    ).exists()
