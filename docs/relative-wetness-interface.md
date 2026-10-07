# Relative wetness and static hydrologic instability

This document specifies the hydrologic stability interface introduced during
development of ShallowLandslider 2.0. It covers the first foundation only:
supplying dynamic wetness to the existing stability equations and identifying
static failure without earthquake shaking. Rainfall-to-recharge conversion,
probabilistic hydrologic selection, event sequencing, and landscape-evolving
runout are separate stages described in
[`hydrologic-triggering-plan.md`](hydrologic-triggering-plan.md).

## Scientific definition

ShallowLandslider represents the hydrologic contribution to pore pressure with
relative wetness, conventionally denoted by `m`:

```text
m = saturated thickness / potentially unstable soil thickness
```

The public range is `0 <= m <= 1`:

- `m = 0` represents the dry/no-saturated-thickness limit;
- `m = 1` represents saturation through the potentially unstable soil column.

This quantity is not necessarily the same as volumetric water content or the
root-zone saturation fraction produced by a soil-moisture model. A hydrology
model must translate its own state into saturated thickness relative to the
failure soil depth before writing `soil__relative_wetness`.

## Configuration contract

One `ShallowLandslider` component supports both fixed and time-varying wetness.
The `wetness_source` argument determines the source.

### Constant wetness

```python
landslider = ShallowLandslider(
    grid,
    submerged_soil_proportion=0.4,
    wetness_source="constant",
)
```

This is the default and preserves the established scalar interface. The scalar
must be finite and within `[0, 1]`.

### Field-driven wetness

```python
relative_wetness = grid.add_zeros("soil__relative_wetness", at="node")
relative_wetness[:] = initial_wetness

landslider = ShallowLandslider(grid, wetness_source="field")
```

Field mode requires `soil__relative_wetness` to exist at nodes when the
component is constructed. Core-node values must be finite and within `[0, 1]`.
Boundary nodes may use NaN as a Landlab boundary sentinel.

The field is read and validated every time stability is evaluated. Another
component can therefore update it without reconstructing ShallowLandslider:

```python
# Dry/background evaluation. No PGA fields are required; omitted PGA is zero.
relative_wetness[:] = dry_state
landslider.run_one_step()
dry_fos = landslider.results["factor_of_safety"].copy()

# A hydrology component updates this same field during a later storm.
relative_wetness[:] = storm_state
landslider.run_one_step()
storm_fos = landslider.results["factor_of_safety"].copy()
```

Invalid updated values raise during the next evaluation. Values are not
silently clipped because clipping would hide coupling, unit, or numerical
errors in the hydrology model.

## Stability equation

The established ShallowLandslider factor-of-safety equation is unchanged:

```text
FoS = (C - m gamma_w h tan(phi)) / (gamma_s h sin(beta))
      + tan(phi) / tan(beta)
```

where:

- `C` is effective cohesion;
- `h` is soil depth;
- `phi` is internal friction angle;
- `beta` is terrain slope;
- `gamma_s` is soil unit weight;
- `gamma_w` is water unit weight;
- `m` is relative wetness.

Field mode changes only whether `m` is spatially and temporally variable. It
does not replace this equation with the SINMAP factor-of-safety equation.

## Critical relative wetness

ShallowLandslider calculates `critical_relative_wetness`, or `m_c`, by solving
its factor-of-safety equation for `m` at `FoS = 1`:

```text
m_c = (C + gamma_s h sin(beta) * (tan(phi) / tan(beta) - 1))
      / (gamma_w h tan(phi))
```

Interpretation:

- `m_c <= 0`: the node is unstable in the dry limit;
- `0 < m_c <= 1`: hydrologic forcing can reach the static failure threshold;
- `m_c > 1`: saturation alone cannot reach the threshold for the current
  material and terrain parameters.

`m_c` is not clipped to `[0, 1]`, because out-of-range values carry this useful
physical meaning. It is available after stability calculation as:

```python
landslider.results["critical_relative_wetness"]
```

The current wetness values used in the same calculation are available as:

```python
landslider.results["relative_wetness"]
```

These are diagnostic arrays in `results`, not additional persistent grid
fields.

## Relationship to critical acceleration

When horizontal and vertical PGA are zero, the existing equations satisfy:

```text
a_critical = g sin(beta) (FoS - 1)
```

The driving acceleration is zero, so the existing instability condition
`a_driving > a_critical` becomes equivalent to `FoS < 1`. This is why static
hydrologic failure can use the same stability calculation rather than a second
trigger equation.

At exactly `FoS = 1`, both sides of the strict instability comparison are zero.
The node becomes unstable after it crosses below one.

## PGA behavior

PGA is no longer synthesized when it is omitted:

- no horizontal PGA argument or field means horizontal PGA is zero;
- no vertical PGA argument or field means vertical PGA is zero.

Earthquake simulations must provide `pga_h`/`pga_v` or populate the existing
`earthquake__horizontal_pga` and `earthquake__vertical_pga` fields. Existing
grid fields remain live references and may be updated by a time-dependent
event driver.

The former `pga_h_max` and `pga_v_max` constructor fallbacks have been removed.
This is an intentional breaking change for the 2.0 interface: a model should
never experience an implicit earthquake merely because PGA was absent.

## Current scope and limitations

This interface makes time-varying static stability possible, but it is not yet
a complete rainfall-landslide model:

- It does not calculate wetness directly from rainfall.
- It does not yet calculate hydrologic selection probability from `m / m_c`.
- It does not provide a hydrologic runout-distance model.
- Existing runout does not yet update topographic elevation and cached terrain
  derivatives consistently for the next event.
- It does not yet provide the earthquake/storm event-sequence driver.

Those capabilities build on this field contract. A steady-state SINMAP-style
adapter or transient groundwater component will write the same
`soil__relative_wetness` field, allowing ShallowLandslider to remain independent
of the particular hydrology model.

## Real-DEM demonstration

The repository includes a reproducible graphical check using a central
subregion of the bundled Nepal SRTM DEM:

```bash
python examples/plot_nepal_hydrologic_foundation.py
```

It evaluates a background state and a prescribed spatial storm-wetness field
on the same component instance with zero PGA. The output panels compare terrain,
soil depth, `m_c`, current `m`, factor of safety, and instability masks. The
default output is `analysis_output/nepal_hydrologic_foundation.png`.

This is deliberately an interface test rather than a hydrologic calibration:
the storm footprint is synthetic. A later wetness adapter will replace that
prescription with state calculated from recharge, transmissivity, contributing
area, or transient groundwater.
