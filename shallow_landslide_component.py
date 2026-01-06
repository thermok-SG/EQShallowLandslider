from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
from landlab.core.model_component import Component

# Integrations from provided helper modules
from helper_functions import (
    # Landslide selection
    generate_landslide_probability,
    probabilistic_group_selection,
    generate_landslide_proportion_from_pga,
    select_groups_by_proportion_weighted,
    # Stats
    recursive_split_wide_regions,
    # Regions
    calculate_regions,
    split_groups_by_aspect,
    _create_zones,
    calculate_region_properties,
    # Newmark
    factor_of_safety,
    critical_transient_acceleration,
    calculate_newmark_displacement,
)


class ShallowLandslider(Component):
    r"""
    Predict shallow landslide initiation & selection on a Landlab grid.

    This component calculates node-wise static stability metrics, identifies
    contiguous unstable regions, optionally splits them by aspect and measured
    dimensions (via KDE), selects potential landslides by either a probabilistic
    or PGA-weighted method, and (optionally) computes Newmark displacement.
    It also exposes a per-group properties table.

    Parameters
    ----------
    grid : landlab.ModelGrid
        A Landlab grid (typically a RasterModelGrid) with required fields.
    cohesion_eff : float
        Effective cohesion (Pa). May be a scalar or array.
    angle_int_frict : float
        Angle of internal friction **in degrees**; converted internally to radians.
    submerged_soil_proportion : float, optional
        Proportion of soil submerged (0–1). Default is 0.0.
    pga_h, pga_v : float or array-like, optional
        Horizontal/vertical PGA in multiples of *g* (if not supplied, fallback
        values are written to grid core nodes).
    pga_h_max, pga_v_max : float, optional
        Fallback PGA values in multiples of *g* to populate core nodes.
    aspect_interval : int, optional
        Degrees per aspect zone for subgrouping (default: 20).
    selection_method : {"probabilistic", "pga_weighted"}, optional
        Landslide selection strategy (default: "probabilistic").
    proportion_method : str, optional
        Method for deriving proportions in probabilistic selection (default: "conservative").
    random_seed : int, optional
        Seed to ensure reproducible selections where randomness is involved.
    time_shaking : float, optional
        Duration (seconds) used by Newmark displacement when `compute_displacement=True`.
    compute_displacement : bool, optional
        If True, compute Newmark displacement on selected labels (default: False).
    displacement_threshold : float, optional
        Threshold (meters) to identify "high displacement" nodes (default: 0.0).
    update_soil : bool, optional
        If True and no soil field exists, add a default `soil__depth` field (default: False).
    g : float, optional
        Gravitational acceleration. Default: 9.81 m s^-2.
    split_by_width_config : dict, optional
        Configuration for recursive KDE-based splitting. Keys may include:
        `{"kde_data", "kde_transform", "width_threshold", "max_iterations",
          "min_region_size", "convergence_threshold"}`.
    verbose : bool, optional
        Print additional progress information.

    Notes
    -----
    **Grid fields written**
    (node-centered; names follow Landlab conventions):

    - ``landslide__factor_of_safety``
    - ``landslide__critical_acceleration``
    - ``landslide__driving_minus_critical_acceleration``
    - ``landslide__unstable_mask`` (bool)
    - ``landslide__region_labels`` (int)
    - ``landslide__aspect_subgroup_labels`` (int)
    - ``landslide__dimension_split_labels`` (int; present only if splitting is configured)
    - ``landslide__selected_labels`` (int)
    - ``landslide__newmark_displacement`` (float; present only if `compute_displacement=True`)

    The ``results`` property returns a dict with cached arrays and a per-group
    properties DataFrame (if computed).

    Examples
    --------
    Basic usage on a tiny grid (doctest):

    >>> import numpy as np
    >>> from landlab import RasterModelGrid
    >>> from helper_functions import generate_acceleration_grid
    >>> from shallow_landslide_component import ShallowLandslider
    >>> mg = RasterModelGrid((5, 5), xy_spacing=10.0)
    >>> z = mg.add_zeros("topographic__elevation", at="node")
    >>> # Create a gentle gradient so slopes are non-zero
    >>> z[:] = np.linspace(0, mg.number_of_nodes - 1, mg.number_of_nodes)
    >>> # Provide soil depth if you intend to update or compute stability with soil terms
    >>> h = mg.add_zeros("soil__depth", at="node"); h[:] = 1.0
    >>> # Provide horizontal/vertical PGA (multiples of g) or let component create fallbacks
    >>> pga_h, pga_v = generate_acceleration_grid(mg, horizontal_max=0.2, vertical_max=0.1,
    ...                                           distribution="uniform")
    >>> comp = ShallowLandslider(
    ...     mg,
    ...     cohesion_eff=500.0,              # Pa
    ...     angle_int_frict=30.0,            # degrees
    ...     pga_h=pga_h, pga_v=pga_v,        # multiples of g
    ...     selection_method="probabilistic",
    ...     compute_displacement=False,
    ...     aspect_interval=45,
    ...     random_seed=42,
    ... )
    >>> comp.run_one_step()
    >>> # The component writes several node fields; check a few are present:
    >>> set(("landslide__factor_of_safety",
    ...      "landslide__critical_acceleration",
    ...      "landslide__unstable_mask")).issubset(mg.at_node.keys())
    True
    >>> # Region labels should be integer valued at nodes:
    >>> labels = mg.at_node["landslide__region_labels"]
    >>> labels.dtype == np.int64 or labels.dtype == np.int32
    True
    >>> # The results dictionary contains cached arrays:
    >>> isinstance(comp.results, dict) and "unstable_mask" in comp.results
    True

    Example with Newmark displacement (doctest):

    >>> comp2 = ShallowLandslider(
    ...     mg,
    ...     cohesion_eff=500.0,
    ...     angle_int_frict=30.0,
    ...     pga_h=pga_h, pga_v=pga_v,
    ...     selection_method="probabilistic",
    ...     compute_displacement=True,
    ...     time_shaking=2.0,                 # seconds
    ...     displacement_threshold=0.0,
    ...     random_seed=123,
    ... )
    >>> comp2.run_one_step()
    >>> "landslide__newmark_displacement" in mg.at_node
    True
    >>> disp = mg.at_node["landslide__newmark_displacement"]
    >>> # Displacement must be non-negative (zero where not selected)
    >>> np.nanmin(disp) >= 0
    True

    """

    _name = "ShallowLandslider"
    _unit_agnostic = True
    _info = {
        # Inputs
        "topographic__elevation": {
            "intent": ("in",),
            "mapping": "node",
            "dtype": float,
            "units": "m",
            "optional": False,
            "doc": "Surface elevation.",
        },
        "soil__depth": {
            "intent": ("in",),
            "mapping": "node",
            "dtype": float,
            "units": "m",
            "optional": True,
            "doc": "Soil thickness (optional).",
        },
        "bedrock__elevation": {
            "intent": ("in",),
            "mapping": "node",
            "dtype": float,
            "units": "m",
            "optional": True,
            "doc": "Bedrock elevation (optional).",
        },
        "hill_flow__receiver_node": {
            "intent": ("in",),
            "mapping": "node",
            "dtype": int,
            "units": "-",
            "optional": True,
            "doc": "Hillslope multi-flow receivers (2D array).",
        },
        "hill_flow__receiver_proportions": {
            "intent": ("in",),
            "mapping": "node",
            "dtype": float,
            "units": "-",
            "optional": True,
            "doc": "Hillslope multi-flow proportions (2D array).",
        },
        "earthquake__horizontal_pga": {
            "intent": ("in",),
            "mapping": "node",
            "dtype": float,
            "units": "g",
            "optional": True,
            "doc": "Horizontal PGA (multiples of g).",
        },
        "earthquake__vertical_pga": {
            "intent": ("in",),
            "mapping": "node",
            "dtype": float,
            "units": "g",
            "optional": True,
            "doc": "Vertical PGA (multiples of g).",
        },
        # Outputs
        "landslide__factor_of_safety": {
            "intent": ("out",),
            "mapping": "node",
            "dtype": float,
            "units": "-",
            "optional": False,
            "doc": "Static factor of safety.",
        },
        "landslide__critical_acceleration": {
            "intent": ("out",),
            "mapping": "node",
            "dtype": float,
            "units": "m s^-2",
            "optional": False,
            "doc": "Critical transient acceleration.",
        },
        "landslide__driving_minus_critical_acceleration": {
            "intent": ("out",),
            "mapping": "node",
            "dtype": float,
            "units": "m s^-2",
            "optional": False,
            "doc": "Driving minus critical acceleration.",
        },
        "landslide__unstable_mask": {
            "intent": ("out",),
            "mapping": "node",
            "dtype": bool,
            "units": "-",
            "optional": False,
            "doc": "Boolean unstable mask.",
        },
        "landslide__region_labels": {
            "intent": ("out",),
            "mapping": "node",
            "dtype": int,
            "units": "-",
            "optional": False,
            "doc": "Failure region labels (0 for none).",
        },
        "landslide__aspect_subgroup_labels": {
            "intent": ("out",),
            "mapping": "node",
            "dtype": int,
            "units": "-",
            "optional": False,
            "doc": "Labels after aspect filtering/splitting.",
        },
        "landslide__dimension_split_labels": {
            "intent": ("out",),
            "mapping": "node",
            "dtype": int,
            "units": "-",
            "optional": True,
            "doc": "Labels after recursive width-based splitting.",
        },
        "landslide__selected_labels": {
            "intent": ("out",),
            "mapping": "node",
            "dtype": int,
            "units": "-",
            "optional": False,
            "doc": "Labels for selected landslides (0 for unselected).",
        },
        "landslide__newmark_displacement": {
            "intent": ("out",),
            "mapping": "node",
            "dtype": float,
            "units": "m",
            "optional": True,
            "doc": "Newmark displacement (optional).",
        },
    }

    def __init__(
        self,
        grid,
        cohesion_eff: float,
        angle_int_frict: float,
        submerged_soil_proportion: float = 0.5,
        pga_h: Optional[np.ndarray | float] = None,
        pga_v: Optional[np.ndarray | float] = None,
        pga_h_max: float = 0.3,
        pga_v_max: float = 0.1,
        aspect_interval: int = 20,
        selection_method: str = "probabilistic",
        proportion_method: str = "conservative",
        random_seed: Optional[int] = None,
        time_shaking: float = 0.0,
        compute_displacement: bool = False,
        displacement_threshold: float = 0.0,
        update_soil: bool = False,
        g: float = 9.81,
        split_by_width_config: Optional[dict] = None,
        verbose: bool = False,
    ):
        super().__init__(grid)
        self.cohesion_eff = float(cohesion_eff)
        self.angle_int_frict = float(np.radians(angle_int_frict))
        self.submerged_soil_proportion = float(submerged_soil_proportion)
        self.aspect_interval = int(aspect_interval)
        self.selection_method = str(selection_method)
        self.proportion_method = str(proportion_method)
        self.random_seed = random_seed
        self.time_shaking = float(time_shaking)
        self.compute_displacement = bool(compute_displacement)
        self.displacement_threshold = float(displacement_threshold)
        self.update_soil = bool(update_soil)
        self.g = float(g)
        self.split_by_width_config = split_by_width_config
        self.verbose = verbose

        # Internals
        self._fos = None
        self._a_transient = None
        self._a_driving = None
        self._a_diff = None
        self._unstable_mask = None
        self._labels = None
        self._aspect_labels = None
        self._split_labels = None
        self._selected_labels = None
        self._selected_proportion = None
        self._newmark = None
        self._high_disp_nodes = None
        self._group_properties_df = None
        self._group_properties_labels = None

        # Ensure optional inputs
        z = self.grid.at_node["topographic__elevation"]
        if "soil__depth" not in self.grid.at_node and self.update_soil:
            h = self.grid.add_zeros("soil__depth", at="node")
            h[:] = 0.5
        if "bedrock__elevation" not in self.grid.at_node:
            br = self.grid.add_zeros("bedrock__elevation", at="node")
            br[:] = z - self.grid.at_node.get("soil__depth", np.zeros_like(z))

        # PGA fields
        self._pga_h = self._get_or_create_pga_field(
            "earthquake__horizontal_pga", pga_h, pga_h_max
        )
        self._pga_v = self._get_or_create_pga_field(
            "earthquake__vertical_pga", pga_v, pga_v_max
        )

        # Cache aspect (as plain ndarray to avoid decorated-array side effects)
        _asp = self.grid.calc_aspect_at_node(
            elevs="topographic__elevation",  # note the double underscore in field name
            unit="degrees",
            ignore_closed_nodes=True,
        )
        self._aspect = np.asarray(_asp, dtype=float).copy()
        self._aspect[self.grid.boundary_nodes] = np.nan

    def _get_or_create_pga_field(
        self, name: str, provided, fallback: float
    ) -> np.ndarray:
        if name in self.grid.at_node:
            return self.grid.at_node[name]
        arr = self.grid.add_zeros(name, at="node")
        if provided is None:
            arr[self.grid.core_nodes] = float(fallback)
        else:
            if np.isscalar(provided):
                arr[self.grid.core_nodes] = float(provided)
            else:
                provided = np.asarray(provided)
                if provided.size != self.grid.number_of_nodes:
                    raise ValueError(
                        f"{name} must be size (n_nodes,), got {provided.size}"
                    )
                arr[:] = provided
        arr[self.grid.boundary_nodes] = np.nan
        return arr

    @property
    def results(self) -> Dict[str, Any]:
        return {
            "factor_of_safety": self._fos,
            "a_transient": self._a_transient,
            "a_driving": self._a_driving,
            "a_diff": self._a_diff,
            "unstable_mask": self._unstable_mask,
            "labels": self._labels,
            "aspect_labels": self._aspect_labels,
            "split_labels": self._split_labels,
            "selected_labels": self._selected_labels,
            "selected_proportion": self._selected_proportion,
            "newmark": self._newmark,
            "high_displacement_nodes": self._high_disp_nodes,
            "group_properties": self._group_properties_df,
        }

    def run_one_step(
        self, dt: Optional[float] = None, kde_input: Optional[dict] = None
    ):
        if kde_input is not None:
            self.split_by_width_config = kde_input
        self._compute_stability()
        self._identify_regions()
        self._filter_by_aspect_and_split()
        self._compute_group_properties()
        self._select_groups()
        if self.compute_displacement:
            self._compute_displacement(dt or self.time_shaking)
        if self.update_soil and self.compute_displacement:
            pass

    def _compute_stability(self):
        self._fos = factor_of_safety(self.grid, self.cohesion_eff, self.angle_int_frict)
        self.grid.at_node["landslide__factor_of_safety"] = self._fos
        a_transient_eq, a_sliding_eq, a_diff_eq = critical_transient_acceleration(
            self.grid,
            self.cohesion_eff,
            self.angle_int_frict,
            submerged_soil_proportion=self.submerged_soil_proportion,
            a_h=self._pga_h * self.g,
            a_v=self._pga_v * self.g,
        )
        self._a_transient = a_transient_eq
        self._a_driving = a_sliding_eq
        self._a_diff = a_diff_eq
        self.grid.at_node["landslide__critical_acceleration"] = self._a_transient
        self.grid.at_node["landslide__driving_minus_critical_acceleration"] = (
            self._a_diff
        )
        unstable = self._a_driving > self._a_transient
        unstable[self.grid.boundary_nodes] = False
        self._unstable_mask = unstable
        self.grid.at_node["landslide__unstable_mask"] = unstable.astype(bool)

    def _identify_regions(self):
        sliding_bool = self._unstable_mask.reshape(self.grid.shape)
        labels, num_features = calculate_regions(sliding_bool, connect_val=8)
        self._labels = labels.reshape(self.grid.number_of_nodes)
        self.grid.at_node["landslide__region_labels"] = self._labels

    def _filter_by_aspect_and_split(self):
        zones = _create_zones(interval=self.aspect_interval)
        aspect_grid = self._aspect.reshape(self.grid.shape)
        aspect_subgroups, aspect_zones, info = split_groups_by_aspect(
            groups=self._labels.reshape(self.grid.shape),
            aspect_array=aspect_grid,
            zones=zones,
            verbose=self.verbose,
        )
        self._aspect_labels = aspect_subgroups.reshape(self.grid.number_of_nodes)
        self.grid.at_node["landslide__aspect_subgroup_labels"] = self._aspect_labels

        # optional recursive split by measured-data KDEs
        if self.split_by_width_config is not None:
            cfg = self.split_by_width_config
            kde_data = cfg.get("kde_data")
            kde_transform = cfg.get("kde_transform")
            width_threshold = cfg.get("width_threshold", 1.5)
            max_iterations = cfg.get("max_iterations", 10)
            min_region_size = cfg.get("min_region_size", 10)
            convergence_threshold = cfg.get("convergence_threshold", 0.75)
            split_labels, split_info = recursive_split_wide_regions(
                grid=self.grid,
                labeled_array=self._aspect_labels.reshape(self.grid.shape),
                aspect_array=self._aspect.reshape(self.grid.shape),
                slopes_grid=np.degrees(
                    self.grid.calc_slope_at_node(elevs="topographic__elevation")
                ).reshape(self.grid.shape),
                kde_results=kde_data,
                transform_info=kde_transform,
                width_threshold=width_threshold,
                max_iterations=max_iterations,
                min_region_size=min_region_size,
                convergence_threshold=convergence_threshold,
                verbose=self.verbose,
            )
            self._split_labels = split_labels.reshape(self.grid.number_of_nodes)
            self.grid.at_node["landslide__dimension_split_labels"] = self._split_labels

    def _compute_group_properties(self):
        """Compute per-group geometric/topographic properties for final subgroups."""
        subgroup_array = (
            self._split_labels.copy()
            if self._split_labels is not None
            else self._aspect_labels.copy()
        )
        slopes_deg = np.degrees(
            self.grid.calc_slope_at_node(elevs="topographic__elevation")
        )
        aspect_grid = self._aspect.reshape(self.grid.shape)

        props_df, working_labels = calculate_region_properties(
            grid=self.grid,
            labeled_array=subgroup_array.reshape(self.grid.shape),
            slopes=slopes_deg,
            aspect_array=aspect_grid,
            min_size=1,
            handle_small="keep",
        )
        self._group_properties_df = props_df
        self._group_properties_labels = working_labels.reshape(
            self.grid.number_of_nodes
        )

    def _select_groups(self):
        subgroup_array = (
            self._split_labels.copy()
            if self._split_labels is not None
            else self._aspect_labels.copy()
        )
        assert (
            subgroup_array.ndim == 1
            and subgroup_array.size == self.grid.number_of_nodes
        )
        if self.selection_method == "probabilistic":
            probs, meta = generate_landslide_probability(
                self.grid,
                h_pga_array=self._pga_h,
                v_pga_array=self._pga_v,
                labeled_array=subgroup_array.reshape(self.grid.shape),  # 2-D for masks
                slope_array=np.degrees(
                    self.grid.calc_slope_at_node(elevs="topographic__elevation")
                ),  # selection.py will reshape internally
                soil_array=None,
                geological_factor_array=None,
                critical_acceleration_array=self._a_transient,  # selection.py will reshape internally
                default_critical_acceleration=0.2,
                random_seed=self.random_seed,
                normalise_final_probs=True,
            )

            selected_groups, meta_sel = probabilistic_group_selection(
                labeled_array=subgroup_array.reshape(self.grid.shape),  # 2-D
                probability_array=probs,  # 2-D
                proportion_method=self.proportion_method,
                custom_proportion=None,
                random_seed=self.random_seed,
                reproducible=True,
            )
            # Flatten before storing/writing to node field
            self._selected_labels = selected_groups.reshape(self.grid.number_of_nodes)
            self._selected_proportion = meta_sel.get("proportion_calculated", None)
        elif self.selection_method == "pga_weighted":
            probs, proportion, meta = generate_landslide_proportion_from_pga(
                self.grid,
                h_pga=self._pga_h,
                v_pga=self._pga_v,
                labeled_array=subgroup_array.reshape(self.grid.shape),  # 2-D
                weight_array=self._a_transient.reshape(self.grid.shape),
                random_seed=self.random_seed,
            )

            groups, labels = select_groups_by_proportion_weighted(
                labeled_array=subgroup_array.reshape(self.grid.shape),  # 2-D
                probability_array=probs,  # 2-D
                proportion=proportion,
            )

            # Flatten before storing/writing to node field
            self._selected_labels = groups.reshape(self.grid.number_of_nodes)

            self._selected_proportion = proportion
        else:
            raise ValueError(f"Unknown selection_method: {self.selection_method}")
        self.grid.at_node["landslide__selected_labels"] = self._selected_labels
        # Annotate selected groups in properties table
        try:
            if self._group_properties_df is not None:
                sel = np.unique(self._selected_labels[self._selected_labels > 0])
                self._group_properties_df["selected"] = (
                    self._group_properties_df.index.isin(sel)
                )
        except Exception:
            pass

    def _compute_displacement(self, time_shaking: float):
        a_diff = self._a_diff.copy()
        a_diff[a_diff < 0] = 0.0
        time_map = (
            np.ones_like(self._selected_labels).reshape(self.grid.shape) * time_shaking
        )
        newmark = calculate_newmark_displacement(
            self.grid,
            a_difference=a_diff,
            filtered_labeled_array=self._selected_labels.reshape(self.grid.shape),
            time_shaking=time_map,
        )
        self._newmark = newmark
        self.grid.at_node["landslide__newmark_displacement"] = newmark
        mask = np.zeros(self.grid.number_of_nodes, dtype=bool)
        mask[newmark > self.displacement_threshold] = True
        self._high_disp_nodes = np.where(mask)[0]

    def export_group_properties(self, path: str):
        """Export the computed group properties to CSV. Run after run_one_step()."""
        if self._group_properties_df is None:
            raise RuntimeError(
                "Group properties not yet computed. Run the component first."
            )
        import os

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._group_properties_df.to_csv(path)
        return path
