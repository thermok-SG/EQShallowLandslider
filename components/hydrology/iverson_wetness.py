"""Transient Iverson (2000) infiltration response for ShallowLandslider.

This module implements the pressure-head solution in equations 27a--27e of
Iverson (2000), *Landslide triggering by rain infiltration*.  It deliberately
implements only hydrology.  Iverson's separate infinite-slope stability and
acceleration equations are not used; :class:`IversonWetnessCoupler` publishes
the existing ``soil__relative_wetness`` field consumed by ShallowLandslider.

For a pressure head ``psi`` evaluated at Iverson depth ``Z`` and the
ShallowLandslider soil-column depth ``h`` (all in metres), the compatibility
conversion is

``soil__relative_wetness = clip(psi / h, 0, 1)``.

ShallowLandslider subsequently reconstructs pore pressure as
``u = soil__relative_wetness * gamma_w * soil__depth``.  Therefore, wherever
``0 <= psi <= h``, the unchanged stability equations receive exactly
``u = gamma_w * psi``.  Negative pressure head (matric suction) is omitted,
as it is in the existing wetness contract, and the beta-line cap prevents the
linear infiltration solution from exceeding its stated shallow-pressure
limit.

The model is a local, one-dimensional pressure-diffusion response.  It has
storm and recession memory but no lateral flow, contributing-area term,
catchment water balance, evapotranspiration, or moving groundwater table.
``infiltration_rate`` is the flux entering the soil surface, not raw rainfall
and not the groundwater recharge consumed by GroundwaterDupuitPercolator.

References
----------
Iverson, R. M. (2000). Landslide triggering by rain infiltration. Water
Resources Research, 36(7), 1897--1910. https://doi.org/10.1029/2000WR900090
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import erfc

from .groundwater_wetness import _node_array


def pressure_response(normalized_time: Any) -> np.ndarray:
    r"""Return Iverson's dimensionless pressure response ``R(t*)``.

    The response is zero at and before ``t*=0`` and, for positive normalized
    time, is

    .. math::

        R(t^*) = \sqrt{t^*/\pi}\exp(-1/t^*)
                 - \operatorname{erfc}(1/\sqrt{t^*}).

    Parameters
    ----------
    normalized_time : scalar or array-like
        Dimensionless time. Values must be finite and non-negative.
    """
    time = np.asarray(normalized_time, dtype=float)
    if not np.all(np.isfinite(time)) or np.any(time < 0.0):
        raise ValueError("normalized_time must be finite and non-negative")

    response = np.zeros_like(time, dtype=float)
    positive = time > 0.0
    scaled = time[positive]
    response[positive] = np.sqrt(scaled / np.pi) * np.exp(
        -1.0 / scaled
    ) - erfc(1.0 / np.sqrt(scaled))
    return response


@dataclass
class _InfiltrationPulse:
    """One constant normalized-infiltration interval."""

    start: float
    end: float
    ratio: np.ndarray


class IversonWetnessCoupler:
    r"""Evolve transient pressure head and publish effective wetness.

    Parameters
    ----------
    grid : landlab.ModelGrid
        Grid containing ``topographic__elevation`` and ``soil__depth``.
    hydraulic_diffusivity : float, array, or field name
        Reference hydraulic diffusivity :math:`D_0` in m2/s. It must be
        positive wherever the failure-plane depth is positive. Iverson's
        slope-adjusted diffusivity is
        :math:`\hat D = 4 D_0 \cos^2\alpha`.
    vertical_hydraulic_conductivity : float, array, or field name
        Vertical saturated hydraulic conductivity :math:`K_z` in m/s. It must
        be positive wherever the failure-plane depth is positive.
    infiltration_rate : float, array, or field name
        Initial surface infiltration flux :math:`I_z` in m/s. This is not raw
        rainfall. Values greater than :math:`K_z` are limited to
        :math:`I_z/K_z=1`; the excess is exposed as ``surface_runoff_rate``.
    initial_water_table_depth : float, array, or field name
        Depth ``d`` to the steady initial water table, measured in the same
        downward-positive coordinate as the failure-plane depth. The default
        ``"soil__depth"`` makes initial pressure zero at the modelled soil
        base. Values must be non-negative.
    steady_infiltration_ratio : float, array, or field name
        Dimensionless steady background ratio entering
        :math:`\beta=\cos^2\alpha-(I_z/K_z)_{steady}`. It must lie between zero
        and :math:`\cos^2\alpha`, inclusive. It defines the initial pressure
        profile but is not appended to the transient storm history.
    failure_depth : float, array, field name, or None
        Depth ``Z`` at which pressure is evaluated. ``None`` uses
        ``soil__depth`` and keeps the hydrology aligned with the failure depth
        assumed by ShallowLandslider.
    slope_angle : float, array, field name, or None
        Slope angle ``alpha`` in radians. ``None`` recalculates slope from
        ``topographic__elevation`` whenever pressure is updated.

    Notes
    -----
    ``run_one_step`` treats its infiltration rate as constant over the supplied
    interval. Piecewise-constant intervals are superposed analytically, so the
    object retains storm and recession history without a numerical diffusion
    timestep. Adjacent intervals with identical forcing are merged.

    The raw solution is retained in ``soil__pressure_head_raw``. The field
    ``soil__pressure_head`` applies Iverson's upper beta-line limit but may
    remain negative. Only the published ``soil__relative_wetness`` discards
    suction and is constrained to ``[0, 1]``. Pressure head is divided by
    ``soil__depth`` (the stability column ``h``), not by a potentially distinct
    Iverson evaluation depth ``Z``. This preserves ``u = gamma_w * psi`` in the
    unchanged ShallowLandslider equations whenever ``0 <= psi <= h``.
    """

    def __init__(
        self,
        grid,
        *,
        hydraulic_diffusivity=1.0e-4,
        vertical_hydraulic_conductivity=1.0e-5,
        infiltration_rate=0.0,
        initial_water_table_depth="soil__depth",
        steady_infiltration_ratio=0.0,
        failure_depth=None,
        slope_angle=None,
    ):
        self.grid = grid
        for field in ("topographic__elevation", "soil__depth"):
            if field not in grid.at_node:
                raise ValueError(f"Iverson coupling requires node field {field!r}")

        self._diffusivity_source = hydraulic_diffusivity
        self._conductivity_source = vertical_hydraulic_conductivity
        self._water_table_source = initial_water_table_depth
        self._steady_ratio_source = steady_infiltration_ratio
        self._failure_depth_source = (
            "soil__depth" if failure_depth is None else failure_depth
        )
        self._slope_source = slope_angle
        self._requested_infiltration = _node_array(
            grid, infiltration_rate, name="infiltration_rate"
        )
        if np.any(self._requested_infiltration < 0.0):
            raise ValueError("infiltration_rate cannot be negative")

        self.elapsed_time = 0.0
        self._pulses: list[_InfiltrationPulse] = []
        self._validate_parameters()

        self.relative_wetness = self._field("soil__relative_wetness")
        self.raw_pressure_head = self._field("soil__pressure_head_raw")
        self.pressure_head = self._field("soil__pressure_head")
        self._equivalent_thickness = self._field("aquifer__thickness")
        self._water_table = self._field("water_table__elevation")
        self._surface_runoff = self._field(
            "average_surface_water__specific_discharge"
        )
        self._update_pressure()
        self._remember_geometry()

    def _field(self, name: str) -> np.ndarray:
        if name in self.grid.at_node:
            return self.grid.at_node[name]
        return self.grid.add_zeros(name, at="node")

    @property
    def infiltration_rate(self) -> np.ndarray:
        """Requested surface infiltration before the conductivity cap (m/s)."""
        return self._requested_infiltration.copy()

    @property
    def saturated_thickness(self) -> np.ndarray:
        """Equivalent pressure-head thickness used by ShallowLandslider (m)."""
        return self._equivalent_thickness

    @property
    def surface_runoff_rate(self) -> np.ndarray:
        """Current infiltration-capacity excess (m/s)."""
        return self._surface_runoff

    @property
    def pulse_count(self) -> int:
        """Number of stored, merged nonzero infiltration intervals."""
        return len(self._pulses)

    def set_infiltration_rate(self, infiltration_rate) -> None:
        """Set the constant infiltration flux for the next interval (m/s)."""
        requested = _node_array(
            self.grid, infiltration_rate, name="infiltration_rate"
        )
        if np.any(requested < 0.0):
            raise ValueError("infiltration_rate cannot be negative")
        self._requested_infiltration = requested

    def run_one_step(self, dt, *, infiltration_rate=None) -> None:
        """Advance by one piecewise-constant infiltration interval.

        ``dt`` is a physical duration in seconds. Unlike GDP, no numerical
        substeps are needed: all prior rectangular pulses are evaluated at the
        new time using analytical superposition.
        """
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and > 0 seconds")
        if self._geometry_changed():
            raise RuntimeError(
                "Iverson geometry changed, or a pressure-history parameter "
                "changed, since the last step. Recreate the coupler or call "
                "reconcile_after_terrain_change(reset_history=True)."
            )
        if infiltration_rate is not None:
            self.set_infiltration_rate(infiltration_rate)

        ratio, _ = self._transient_ratio_and_excess()
        start = self.elapsed_time
        end = start + dt
        if np.any(ratio > 0.0):
            if self._pulses and np.array_equal(self._pulses[-1].ratio, ratio):
                self._pulses[-1].end = end
            else:
                self._pulses.append(_InfiltrationPulse(start, end, ratio.copy()))
        self.elapsed_time = end
        self._update_pressure()

    def update_runoff_field(self, additional_runoff_rate=0.0) -> np.ndarray:
        """Write infiltration excess plus any external runoff to the grid."""
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
            runoff = self.grid.add_field(
                "water__unit_flux_in", total, at="node", copy=True
            )
        return runoff

    def reconcile_after_terrain_change(self, *, reset_history=False) -> np.ndarray:
        """Acknowledge changed geometry, optionally discarding transient history.

        The analytical pulse history is tied to the old failure depth through
        ``t*=t*Dhat/Z**2`` and cannot be conservatively remapped after erosion or
        deposition. Callers must explicitly choose ``reset_history=True``;
        otherwise this method raises rather than silently changing past storm
        response. A reset returns to the configured steady initial profile at
        elapsed time zero.
        """
        if not reset_history:
            raise RuntimeError(
                "Iverson pressure history cannot be remapped to changed failure "
                "depth. Pass reset_history=True to discard transient history."
            )
        self._validate_parameters()
        self.elapsed_time = 0.0
        self._pulses.clear()
        self._update_pressure()
        self._remember_geometry()
        return self.relative_wetness

    def _depth(self) -> np.ndarray:
        depth = _node_array(
            self.grid, self._failure_depth_source, name="failure_depth"
        )
        if np.any(depth < 0.0):
            raise ValueError("failure_depth cannot be negative")
        return depth

    def _slope(self) -> np.ndarray:
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

    def _parameters(self):
        depth = self._depth()
        slope = self._slope()
        diffusivity = _node_array(
            self.grid, self._diffusivity_source, name="hydraulic_diffusivity"
        )
        conductivity = _node_array(
            self.grid,
            self._conductivity_source,
            name="vertical_hydraulic_conductivity",
        )
        water_table_depth = _node_array(
            self.grid,
            self._water_table_source,
            name="initial_water_table_depth",
        )
        steady_ratio = _node_array(
            self.grid,
            self._steady_ratio_source,
            name="steady_infiltration_ratio",
        )
        active = depth > 0.0
        if np.any(diffusivity[active] <= 0.0):
            raise ValueError(
                "hydraulic_diffusivity must be > 0 where failure_depth is positive"
            )
        if np.any(conductivity[active] <= 0.0):
            raise ValueError(
                "vertical_hydraulic_conductivity must be > 0 where failure_depth "
                "is positive"
            )
        if np.any(water_table_depth < 0.0):
            raise ValueError("initial_water_table_depth cannot be negative")
        beta = np.cos(slope) ** 2 - steady_ratio
        if np.any(steady_ratio < 0.0) or np.any(beta < -1.0e-14):
            raise ValueError(
                "steady_infiltration_ratio must lie between 0 and cos(slope)**2"
            )
        beta = np.maximum(beta, 0.0)
        return depth, slope, diffusivity, conductivity, water_table_depth, beta

    def _validate_parameters(self) -> None:
        self._parameters()

    def _transient_ratio_and_excess(self):
        conductivity = _node_array(
            self.grid,
            self._conductivity_source,
            name="vertical_hydraulic_conductivity",
        )
        depth = self._depth()
        active = depth > 0.0
        if np.any(conductivity[active] <= 0.0):
            raise ValueError(
                "vertical_hydraulic_conductivity must be > 0 where failure_depth "
                "is positive"
            )
        ratio = np.zeros(self.grid.number_of_nodes, dtype=float)
        ratio[active] = np.minimum(
            self._requested_infiltration[active] / conductivity[active], 1.0
        )
        excess = np.zeros_like(ratio)
        excess[active] = np.maximum(
            self._requested_infiltration[active] - conductivity[active], 0.0
        )
        return ratio, excess

    def _update_pressure(self) -> None:
        depth, slope, diffusivity, _, water_table_depth, beta = self._parameters()
        active = depth > 0.0
        raw = beta * (depth - water_table_depth)
        transient = np.zeros_like(raw)
        d_hat = 4.0 * diffusivity * np.cos(slope) ** 2

        for pulse in self._pulses:
            age_at_start = self.elapsed_time - pulse.start
            age_at_end = np.maximum(self.elapsed_time - pulse.end, 0.0)
            start_star = np.zeros_like(depth)
            end_star = np.zeros_like(depth)
            start_star[active] = age_at_start * d_hat[active] / depth[active] ** 2
            end_star[active] = age_at_end * d_hat[active] / depth[active] ** 2
            transient[active] += depth[active] * pulse.ratio[active] * (
                pressure_response(start_star[active])
                - pressure_response(end_star[active])
            )

        raw += transient
        raw[~active] = 0.0
        beta_limit = beta * depth
        capped = np.minimum(raw, beta_limit)
        capped[~active] = 0.0

        self.raw_pressure_head[:] = raw
        self.pressure_head[:] = capped
        stability_depth = np.asarray(
            self.grid.at_node["soil__depth"], dtype=float
        )
        stability_active = stability_depth > 0.0
        self.relative_wetness[:] = 0.0
        self.relative_wetness[stability_active] = np.clip(
            capped[stability_active] / stability_depth[stability_active],
            0.0,
            1.0,
        )
        self._equivalent_thickness[:] = self.relative_wetness * stability_depth
        self._water_table[:] = (
            np.asarray(self.grid.at_node["topographic__elevation"], dtype=float)
            - stability_depth
            + self._equivalent_thickness
        )
        _, excess = self._transient_ratio_and_excess()
        self._surface_runoff[:] = excess

    def _remember_geometry(self) -> None:
        self._surface_snapshot = self.grid.at_node["topographic__elevation"].copy()
        self._soil_snapshot = self.grid.at_node["soil__depth"].copy()
        (
            depth,
            slope,
            diffusivity,
            _,
            water_table_depth,
            beta,
        ) = self._parameters()
        self._depth_snapshot = depth
        self._slope_snapshot = slope
        self._diffusivity_snapshot = diffusivity
        self._water_table_depth_snapshot = water_table_depth
        self._beta_snapshot = beta

    def _geometry_changed(self) -> bool:
        unchanged = (
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
        )
        if not unchanged:
            return True
        (
            depth,
            slope,
            diffusivity,
            _,
            water_table_depth,
            beta,
        ) = self._parameters()
        return not all(
            np.array_equal(old, new, equal_nan=True)
            for old, new in (
                (self._depth_snapshot, depth),
                (self._slope_snapshot, slope),
                (self._diffusivity_snapshot, diffusivity),
                (self._water_table_depth_snapshot, water_table_depth),
                (self._beta_snapshot, beta),
            )
        )
