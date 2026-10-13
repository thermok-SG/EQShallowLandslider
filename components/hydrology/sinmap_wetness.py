"""SINMAP-style equilibrium wetness for ShallowLandslider.

This module implements only the steady-state hydrologic index used by SINMAP,
not SINMAP's separate stability equation.  It publishes the same
``soil__relative_wetness`` contract as :class:`GroundwaterWetnessCoupler`, so
both hydrology choices drive the existing ShallowLandslider equations.

The equilibrium estimate is

``m = min((R / T) * (a / sin(beta)), 1)``,

where ``R`` is recharge (m/s), ``T`` is soil transmissivity (m2/s), ``a`` is
specific contributing area (m), and ``beta`` is slope angle (radians).  Unlike
GDP, this model has no independent storage state: changing recharge changes
wetness immediately, and zero recharge produces zero wetness.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .groundwater_wetness import _node_array


class SinmapWetnessCoupler:
    """Calculate steady-state, topographically controlled relative wetness.

    Parameters
    ----------
    grid : landlab.ModelGrid
        Grid containing ``topographic__elevation`` and ``soil__depth``.
    transmissivity : float, array, or field name
        Depth-integrated saturated hydraulic conductivity, in m2/s. Values
        must be positive wherever soil depth is positive.
    recharge_rate : float, array, or field name
        Steady groundwater recharge, in m/s.
    specific_contributing_area : float, array, field name, or None
        Upslope contributing area per unit contour width, in m. If omitted,
        ``drainage_area / grid.dx`` is used on raster grids. This approximation
        should be replaced by an explicitly calculated field where contour
        width is important.
    slope_angle : float, array, field name, or None
        Surface slope angle in radians. If omitted, it is recalculated from
        ``topographic__elevation`` with ``grid.calc_slope_at_node``.

    Notes
    -----
    ``run_one_step`` accepts a timestep for interface compatibility with GDP,
    but the timestep does not enter the equilibrium calculation.
    """

    def __init__(
        self,
        grid,
        *,
        transmissivity=1.0e-5,
        recharge_rate=0.0,
        specific_contributing_area=None,
        slope_angle=None,
    ):
        self.grid = grid
        for field in ("topographic__elevation", "soil__depth"):
            if field not in grid.at_node:
                raise ValueError(f"SINMAP wetness requires node field {field!r}")

        self._transmissivity_source = transmissivity
        self._area_source = specific_contributing_area
        self._slope_source = slope_angle
        self._requested_recharge = _node_array(
            grid, recharge_rate, name="recharge_rate"
        )
        if np.any(self._requested_recharge < 0.0):
            raise ValueError("recharge_rate cannot be negative")

        if "soil__relative_wetness" in grid.at_node:
            self.relative_wetness = grid.at_node["soil__relative_wetness"]
        else:
            self.relative_wetness = grid.add_zeros(
                "soil__relative_wetness", at="node"
            )
        if "aquifer__thickness" in grid.at_node:
            self._saturated_thickness = grid.at_node["aquifer__thickness"]
        else:
            self._saturated_thickness = grid.add_zeros(
                "aquifer__thickness", at="node"
            )
        if "water_table__elevation" in grid.at_node:
            self._water_table = grid.at_node["water_table__elevation"]
        else:
            self._water_table = grid.add_zeros(
                "water_table__elevation", at="node"
            )
        if "average_surface_water__specific_discharge" in grid.at_node:
            self._surface_runoff = grid.at_node[
                "average_surface_water__specific_discharge"
            ]
        else:
            self._surface_runoff = grid.add_zeros(
                "average_surface_water__specific_discharge", at="node"
            )

        self._update_equilibrium()

    @property
    def recharge_rate(self) -> np.ndarray:
        """Requested steady recharge rate (m/s)."""
        return self._requested_recharge.copy()

    @property
    def saturated_thickness(self) -> np.ndarray:
        """Equivalent saturated thickness implied by equilibrium wetness (m)."""
        return self._saturated_thickness

    @property
    def surface_runoff_rate(self) -> np.ndarray:
        """Zero runoff field; the SINMAP index does not solve surface runoff."""
        return self._surface_runoff

    def set_recharge_rate(self, recharge_rate) -> None:
        """Set steady recharge for the next equilibrium calculation (m/s)."""
        requested = _node_array(self.grid, recharge_rate, name="recharge_rate")
        if np.any(requested < 0.0):
            raise ValueError("recharge_rate cannot be negative")
        self._requested_recharge = requested

    def run_one_step(self, dt, *, recharge_rate=None) -> None:
        """Refresh equilibrium wetness; *dt* is accepted but has no memory."""
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and > 0 seconds")
        if recharge_rate is not None:
            self.set_recharge_rate(recharge_rate)
        self._update_equilibrium()

    def update_runoff_field(self, additional_runoff_rate=0.0) -> np.ndarray:
        """Write externally supplied runoff to ``water__unit_flux_in``.

        SINMAP wetness diagnoses equilibrium saturation but does not calculate
        a water balance or saturation-excess runoff. Consequently only the
        explicitly supplied runoff is routed.
        """
        additional = _node_array(
            self.grid, additional_runoff_rate, name="additional_runoff_rate"
        )
        if np.any(additional < 0.0):
            raise ValueError("additional_runoff_rate cannot be negative")
        if "water__unit_flux_in" in self.grid.at_node:
            runoff = self.grid.at_node["water__unit_flux_in"]
            runoff[:] = additional
        else:
            runoff = self.grid.add_field(
                "water__unit_flux_in", additional, at="node", copy=True
            )
        return runoff

    def reconcile_after_terrain_change(self) -> np.ndarray:
        """Recalculate equilibrium wetness after routing/terrain fields change."""
        self._update_equilibrium()
        return self.relative_wetness

    def _specific_contributing_area(self) -> np.ndarray:
        if self._area_source is not None:
            area = _node_array(
                self.grid,
                self._area_source,
                name="specific_contributing_area",
            )
        else:
            if "drainage_area" not in self.grid.at_node:
                raise ValueError(
                    "specific_contributing_area is required unless the grid has "
                    "a drainage_area field"
                )
            if not hasattr(self.grid, "dx"):
                raise ValueError(
                    "automatic specific contributing area requires a raster grid"
                )
            area = np.asarray(self.grid.at_node["drainage_area"], dtype=float) / float(
                self.grid.dx
            )
        if np.any(area < 0.0):
            raise ValueError("specific_contributing_area cannot be negative")
        return area

    def _slope_angle(self) -> np.ndarray:
        if self._slope_source is None:
            slope = np.asarray(
                self.grid.calc_slope_at_node(elevs="topographic__elevation"),
                dtype=float,
            )
        else:
            slope = _node_array(self.grid, self._slope_source, name="slope_angle")
        if not np.all(np.isfinite(slope)):
            raise ValueError("slope_angle must contain only finite values")
        if np.any((slope < 0.0) | (slope >= 0.5 * np.pi)):
            raise ValueError("slope_angle must lie within [0, pi/2) radians")
        return slope

    def _update_equilibrium(self) -> None:
        soil = np.asarray(self.grid.at_node["soil__depth"], dtype=float)
        surface = np.asarray(
            self.grid.at_node["topographic__elevation"], dtype=float
        )
        if not np.all(np.isfinite(soil)) or np.any(soil < 0.0):
            raise ValueError("soil__depth must be finite and non-negative")
        if not np.all(np.isfinite(surface)):
            raise ValueError("topographic__elevation must contain only finite values")

        transmissivity = _node_array(
            self.grid, self._transmissivity_source, name="transmissivity"
        )
        active = soil > 0.0
        if np.any(transmissivity[active] <= 0.0):
            raise ValueError("transmissivity must be > 0 where soil depth is positive")

        area = self._specific_contributing_area()
        slope = self._slope_angle()
        water_per_width = self._requested_recharge * area
        capacity_per_width = transmissivity * np.sin(slope)

        wetness = np.zeros(self.grid.number_of_nodes, dtype=float)
        regular = active & (capacity_per_width > 0.0)
        wetness[regular] = water_per_width[regular] / capacity_per_width[regular]
        wetness[active & (capacity_per_width <= 0.0) & (water_per_width > 0.0)] = 1.0
        np.clip(wetness, 0.0, 1.0, out=wetness)

        self.relative_wetness[:] = wetness
        self._saturated_thickness[:] = wetness * soil
        self._water_table[:] = surface - soil + self._saturated_thickness
        self._surface_runoff[:] = 0.0
