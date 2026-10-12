"""Couple transient shallow groundwater to ShallowLandslider wetness.

This module deliberately separates hydrologic state evolution from slope
stability. :class:`GroundwaterWetnessCoupler` composes Landlab's
``GroundwaterDupuitPercolator`` (GDP) with the field contract consumed by
``ShallowLandslider``. It does not calculate rainfall infiltration: its input
is groundwater recharge, expressed as a water flux in metres per second.

The principal conversion is

``soil__relative_wetness = aquifer__thickness / soil__depth``.

Thus the relative-wetness field represents the fraction of the potentially
unstable soil column occupied by the saturated zone. It is not root-zone
volumetric water content.

GDP assumes a positive permeable thickness. Cells with no modelled soil are
given a numerically negligible aquifer thickness internally, receive no
recharge, and are assigned zero relative wetness. The physical soil-depth
field and topographic surface are not changed by this regularisation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from landlab.components import GroundwaterDupuitPercolator


def _node_array(grid, value: Any, *, name: str) -> np.ndarray:
    """Return *value* as a finite float array with one value per node."""
    if isinstance(value, str):
        if value not in grid.at_node:
            raise ValueError(f"node field {value!r} supplied for {name} does not exist")
        array = np.asarray(grid.at_node[value], dtype=float)
    elif np.isscalar(value):
        array = np.full(grid.number_of_nodes, float(value), dtype=float)
    else:
        array = np.asarray(value, dtype=float)

    if array.shape != (grid.number_of_nodes,):
        raise ValueError(
            f"{name} must be a scalar, node-field name, or an array of length "
            f"{grid.number_of_nodes}; got shape {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


class GroundwaterWetnessCoupler:
    """Advance shallow groundwater and publish landslide relative wetness.

    Parameters
    ----------
    grid : landlab.ModelGrid
        Grid containing ``topographic__elevation``, ``soil__depth``, and the
        field named by *aquifer_base_source*.
    hydraulic_conductivity : float, array, field name, or callable
        Saturated hydraulic conductivity passed to GDP, in metres per second.
    porosity : float, array, or field name
        Drainable porosity passed to GDP.
    recharge_rate : float, array, or field name
        Initial groundwater recharge in metres per second. Recharge is forced
        to zero where ``soil__depth`` is zero.
    initial_relative_wetness : float, array, or field name
        Initial saturated thickness divided by soil depth. Values must lie in
        the closed interval [0, 1].
    aquifer_base_source : str
        Node field defining the physical base of the shallow aquifer. For a
        perched colluvial aquifer this is normally ``bedrock__elevation``.
    minimum_aquifer_thickness : float
        Positive numerical thickness used only where the physical aquifer has
        zero thickness, avoiding division by zero inside GDP.
    regularization_f, courant_coefficient, vn_coefficient : float
        Numerical parameters passed unchanged to GDP.

    Notes
    -----
    The coupler treats topography and soil depth as persistent external state.
    If either changes, :meth:`run_one_step` raises until
    :meth:`reconcile_after_terrain_change` has been called. This prevents GDP
    from advancing with a stale aquifer geometry after SPACE or landslide
    runout modifies the landscape.
    """

    def __init__(
        self,
        grid,
        *,
        hydraulic_conductivity=1.0e-5,
        porosity=0.2,
        recharge_rate=0.0,
        initial_relative_wetness=0.0,
        aquifer_base_source="bedrock__elevation",
        minimum_aquifer_thickness=1.0e-6,
        regularization_f=0.01,
        courant_coefficient=0.5,
        vn_coefficient=0.8,
    ):
        self.grid = grid
        self.aquifer_base_source = aquifer_base_source
        self.minimum_aquifer_thickness = float(minimum_aquifer_thickness)
        if not np.isfinite(self.minimum_aquifer_thickness) or (
            self.minimum_aquifer_thickness <= 0.0
        ):
            raise ValueError("minimum_aquifer_thickness must be finite and > 0")

        for field in (
            "topographic__elevation",
            "soil__depth",
            self.aquifer_base_source,
        ):
            if field not in grid.at_node:
                raise ValueError(f"Groundwater coupling requires node field {field!r}")

        self._validate_geometry(require_soil_identity=True)
        initial_wetness = _node_array(
            grid, initial_relative_wetness, name="initial_relative_wetness"
        )
        if np.any((initial_wetness < 0.0) | (initial_wetness > 1.0)):
            raise ValueError("initial_relative_wetness must lie within [0, 1]")

        if "aquifer_base__elevation" in grid.at_node:
            self._aquifer_base = grid.at_node["aquifer_base__elevation"]
        else:
            self._aquifer_base = grid.add_zeros("aquifer_base__elevation", at="node")
        self._sync_aquifer_base()

        initial_thickness = initial_wetness * grid.at_node["soil__depth"]
        initial_water_table = self._aquifer_base + initial_thickness
        if "water_table__elevation" in grid.at_node:
            grid.at_node["water_table__elevation"][:] = initial_water_table
        else:
            grid.add_field("water_table__elevation", initial_water_table, at="node")

        if "soil__relative_wetness" not in grid.at_node:
            grid.add_zeros("soil__relative_wetness", at="node")
        self.relative_wetness = grid.at_node["soil__relative_wetness"]

        if "hydrology__geomorphic_water_excess_depth" not in grid.at_node:
            grid.add_zeros("hydrology__geomorphic_water_excess_depth", at="node")
        self.geomorphic_water_excess = grid.at_node[
            "hydrology__geomorphic_water_excess_depth"
        ]

        self._requested_recharge = _node_array(
            grid, recharge_rate, name="recharge_rate"
        )
        if np.any(self._requested_recharge < 0.0):
            raise ValueError("recharge_rate cannot be negative")
        effective_recharge = self._effective_recharge()

        self.groundwater = GroundwaterDupuitPercolator(
            grid,
            hydraulic_conductivity=hydraulic_conductivity,
            porosity=porosity,
            recharge_rate=effective_recharge,
            regularization_f=regularization_f,
            courant_coefficient=courant_coefficient,
            vn_coefficient=vn_coefficient,
        )

        # GDP assigns a sentinel water-table value at closed boundaries. Keep
        # all three state fields mutually consistent before exposing them.
        thickness = grid.at_node["aquifer__thickness"]
        thickness[:] = np.maximum(initial_thickness, 0.0)
        grid.at_node["water_table__elevation"][:] = self._aquifer_base + thickness
        self._update_relative_wetness()
        self._remember_geometry()

    @property
    def recharge_rate(self) -> np.ndarray:
        """Requested recharge rate before zero-soil masking (m/s)."""
        return self._requested_recharge.copy()

    @property
    def saturated_thickness(self) -> np.ndarray:
        """Live GDP saturated-zone thickness field (m)."""
        return self.grid.at_node["aquifer__thickness"]

    @property
    def surface_runoff_rate(self) -> np.ndarray:
        """GDP interval-mean saturation-excess runoff rate (m/s)."""
        return self.grid.at_node["average_surface_water__specific_discharge"]

    def set_recharge_rate(self, recharge_rate) -> None:
        """Set spatial recharge for subsequent groundwater steps (m/s)."""
        requested = _node_array(self.grid, recharge_rate, name="recharge_rate")
        if np.any(requested < 0.0):
            raise ValueError("recharge_rate cannot be negative")
        self._requested_recharge = requested
        self.groundwater.recharge = self._effective_recharge()

    def run_one_step(self, dt, *, recharge_rate=None) -> None:
        """Advance GDP by *dt* seconds and refresh relative wetness.

        GDP's adaptive solver is always used because its stable explicit
        timestep depends on the evolving saturated thickness and hydraulic
        gradients.
        """
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and > 0 seconds")
        if self._geometry_changed():
            raise RuntimeError(
                "Terrain or aquifer geometry changed since the last hydrology "
                "step. Call reconcile_after_terrain_change() before advancing GDP."
            )
        if recharge_rate is not None:
            self.set_recharge_rate(recharge_rate)
        else:
            self.groundwater.recharge = self._effective_recharge()

        self.groundwater.run_with_adaptive_time_step_solver(dt)
        self._update_relative_wetness()

    def update_runoff_field(self, additional_runoff_rate=0.0) -> np.ndarray:
        """Write total surface runoff to ``water__unit_flux_in``.

        Parameters
        ----------
        additional_runoff_rate : float, array, or field name
            Infiltration-excess or other surface runoff in metres per second.
            Raw rainfall should not be supplied unless all of it truly becomes
            runoff.

        Returns
        -------
        ndarray
            The live ``water__unit_flux_in`` field.
        """
        additional = _node_array(
            self.grid, additional_runoff_rate, name="additional_runoff_rate"
        )
        if np.any(additional < 0.0):
            raise ValueError("additional_runoff_rate cannot be negative")
        total = self.surface_runoff_rate + additional
        if "water__unit_flux_in" in self.grid.at_node:
            runoff = self.grid.at_node["water__unit_flux_in"]
            runoff[:] = total
        else:
            runoff = self.grid.add_field("water__unit_flux_in", total, at="node")
        return runoff

    def reconcile_after_terrain_change(self) -> np.ndarray:
        """Reconcile groundwater storage after erosion or deposition.

        Existing saturated thickness is retained where the new aquifer can
        contain it. If erosion reduces aquifer capacity below the stored
        saturated thickness, the excess is removed and reported as an
        equivalent water depth in
        ``hydrology__geomorphic_water_excess_depth``. Deposited pore space is
        initially dry: deposition increases capacity but does not create water.

        Returns
        -------
        ndarray
            Live node field containing expelled water depth (m). To route this
            water, divide it by the surface-routing interval and add that rate
            through :meth:`update_runoff_field`.
        """
        self._validate_geometry(require_soil_identity=True)
        old_thickness = self.saturated_thickness.copy()
        self._sync_aquifer_base()

        physical_capacity = np.maximum(
            self.grid.at_node["topographic__elevation"]
            - np.asarray(self.grid.at_node[self.aquifer_base_source]),
            0.0,
        )
        new_thickness = np.minimum(old_thickness, physical_capacity)
        new_thickness[self.grid.at_node["soil__depth"] <= 0.0] = 0.0

        porosity = np.asarray(self.groundwater.n, dtype=float)
        self.geomorphic_water_excess[:] = porosity * np.maximum(
            old_thickness - new_thickness, 0.0
        )
        self.saturated_thickness[:] = new_thickness
        self.grid.at_node["water_table__elevation"][:] = (
            self._aquifer_base + new_thickness
        )
        self.groundwater.recharge = self._effective_recharge()
        self._update_relative_wetness()
        self._remember_geometry()
        return self.geomorphic_water_excess

    def _effective_recharge(self) -> np.ndarray:
        recharge = self._requested_recharge.copy()
        recharge[self.grid.at_node["soil__depth"] <= 0.0] = 0.0
        return recharge

    def _sync_aquifer_base(self) -> None:
        physical_base = np.asarray(
            self.grid.at_node[self.aquifer_base_source], dtype=float
        )
        surface = np.asarray(self.grid.at_node["topographic__elevation"], dtype=float)
        capacity = surface - physical_base
        self._aquifer_base[:] = physical_base
        zero_capacity = capacity < self.minimum_aquifer_thickness
        self._aquifer_base[zero_capacity] = (
            surface[zero_capacity] - self.minimum_aquifer_thickness
        )

    def _update_relative_wetness(self) -> None:
        soil = np.asarray(self.grid.at_node["soil__depth"], dtype=float)
        saturated = np.asarray(self.saturated_thickness, dtype=float)
        positive_soil = soil > 0.0
        tolerance = 1.0e-10 * np.maximum(soil, 1.0)
        invalid = positive_soil & (
            (saturated < -tolerance) | (saturated > soil + tolerance)
        )
        if np.any(invalid):
            node = int(np.flatnonzero(invalid)[0])
            raise RuntimeError(
                "GDP saturated thickness lies outside the modelled soil column "
                f"at node {node}: thickness={saturated[node]:.6g} m, "
                f"soil_depth={soil[node]:.6g} m"
            )

        self.relative_wetness[:] = 0.0
        self.relative_wetness[positive_soil] = (
            saturated[positive_soil] / soil[positive_soil]
        )
        # Only remove roundoff after checking for meaningful violations.
        np.clip(self.relative_wetness, 0.0, 1.0, out=self.relative_wetness)

    def _validate_geometry(self, *, require_soil_identity: bool) -> None:
        surface = np.asarray(self.grid.at_node["topographic__elevation"], dtype=float)
        soil = np.asarray(self.grid.at_node["soil__depth"], dtype=float)
        base = np.asarray(self.grid.at_node[self.aquifer_base_source], dtype=float)
        for name, values in (
            ("topographic__elevation", surface),
            ("soil__depth", soil),
            (self.aquifer_base_source, base),
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain only finite values")
        if np.any(soil < 0.0):
            raise ValueError("soil__depth cannot be negative")
        if np.any(base > surface + 1.0e-10):
            raise ValueError("aquifer base cannot be above topographic elevation")
        if require_soil_identity and self.aquifer_base_source == "bedrock__elevation":
            if not np.allclose(surface, base + soil, rtol=0.0, atol=1.0e-8):
                raise ValueError(
                    "Groundwater coupling requires topographic__elevation = "
                    "bedrock__elevation + soil__depth"
                )

    def _remember_geometry(self) -> None:
        self._surface_snapshot = self.grid.at_node["topographic__elevation"].copy()
        self._soil_snapshot = self.grid.at_node["soil__depth"].copy()
        self._base_snapshot = self.grid.at_node[self.aquifer_base_source].copy()

    def _geometry_changed(self) -> bool:
        return not (
            np.array_equal(
                self._surface_snapshot,
                self.grid.at_node["topographic__elevation"],
                equal_nan=True,
            )
            and np.array_equal(
                self._soil_snapshot,
                self.grid.at_node["soil__depth"],
                equal_nan=True,
            )
            and np.array_equal(
                self._base_snapshot,
                self.grid.at_node[self.aquifer_base_source],
                equal_nan=True,
            )
        )
