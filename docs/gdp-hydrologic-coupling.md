# Transient groundwater coupling

## Scope

`GroundwaterWetnessCoupler` connects Landlab's
`GroundwaterDupuitPercolator` (GDP) to the live
`soil__relative_wetness` field consumed by `ShallowLandslider`. This is the
transient-groundwater foundation. It accepts groundwater recharge rather than
rainfall; interception, infiltration, unsaturated storage, evapotranspiration,
and deep drainage remain the responsibility of a later rainfall–recharge
adapter.

The coupling does not change the ShallowLandslider stability equation,
candidate geometry, selection, or multiflow runout calculation.

## State and units

The coupled model uses metres and seconds for hydrology:

| Quantity | Field or argument | Units | Role |
| --- | --- | --- | --- |
| Surface elevation | `topographic__elevation` | m | Upper aquifer boundary |
| Physical aquifer base | `bedrock__elevation` by default | m | Impermeable base or failure-plane elevation |
| Saturated thickness | `aquifer__thickness` | m | Persistent GDP groundwater storage state |
| Water-table elevation | `water_table__elevation` | m | Aquifer base plus saturated thickness |
| Recharge | `recharge_rate` | m s-1 | Water entering the shallow saturated zone |
| Saturation-excess runoff | `average_surface_water__specific_discharge` | m s-1 | GDP surface return flow averaged over its adaptive substeps |
| Routed local runoff | `water__unit_flux_in` | m s-1 | Input to `PriorityFloodFlowRouter` |
| Landslide wetness | `soil__relative_wetness` | dimensionless | Saturated fraction of potentially unstable soil |

The coupling computes

$$
m_i = \frac{h_{s,i}}{h_{f,i}},
$$

where $h_s$ is GDP saturated thickness and $h_f$ is modelled soil depth. A
zero-soil node has `m = 0` and receives zero recharge. Meaningful groundwater
thickness outside `[0, soil__depth]` raises an error; only floating-point
roundoff is clipped.

GDP cannot numerically operate with exactly zero regolith thickness. At
zero-soil nodes the coupler therefore places the internal
`aquifer_base__elevation` a configurable, negligible distance below the
surface. This does not alter `bedrock__elevation`, `soil__depth`, or
`topographic__elevation`, and these nodes remain hydrologically dry.

## Initialisation

For a colluvial aquifer extending to bedrock, initialise mutually consistent
soil and bedrock fields before constructing the coupling:

```python
grid.at_node["bedrock__elevation"] = (
    grid.at_node["topographic__elevation"]
    - grid.at_node["soil__depth"]
)

hydrology = GroundwaterWetnessCoupler(
    grid,
    hydraulic_conductivity=1.0e-5,  # m/s
    porosity=0.25,
    recharge_rate=0.0,              # m/s
    initial_relative_wetness=0.1,
)
```

If the relevant perched aquifer is bounded by a shallower failure plane, use a
different node field as `aquifer_base_source`. In that case the user is
responsible for ensuring that the aquifer thickness and ShallowLandslider soil
depth have a physically meaningful relationship.

## Time stepping and runoff transfer

GDP is always advanced with its adaptive solver:

```python
hydrology.run_one_step(
    storm_duration_seconds,
    recharge_rate=recharge_m_per_s,
)
```

After the call, GDP state and `soil__relative_wetness` describe the end of the
interval. `average_surface_water__specific_discharge` describes mean
saturation-excess runoff over the interval. Before routing surface water:

```python
hydrology.update_runoff_field(infiltration_excess_runoff_m_per_s)
router.run_one_step()
```

This writes

$$
\texttt{water\_\_unit\_flux\_in}
= q_{\mathrm{saturation\ excess}} + q_{\mathrm{additional\ runoff}}.
$$

Raw rainfall must not be used as the additional runoff term unless the model
explicitly assumes that all rainfall becomes surface runoff.

## Landscape change

SPACE and landslide runout can modify soil, bedrock, and topography. The
coupling snapshots this geometry and refuses to advance GDP after a change
until the driver explicitly calls:

```python
expelled_water_depth = hydrology.reconcile_after_terrain_change()
```

The reconciliation uses the following initial policy:

- saturated thickness is conserved where it fits within the new aquifer;
- newly deposited pore space is initially dry and therefore creates no water;
- erosion that cuts below the former water table removes the excess storage;
- removed storage is reported as equivalent water depth in
  `hydrology__geomorphic_water_excess_depth`;
- water-table elevation and relative wetness are refreshed against the new
  geometry.

To route expelled water during a surface interval of duration $\Delta t$,
convert the depth to a rate and add it to the other surface runoff:

```python
geomorphic_runoff_rate = expelled_water_depth / surface_interval_seconds
hydrology.update_runoff_field(
    infiltration_excess_runoff + geomorphic_runoff_rate
)
```

This explicit policy avoids creating water when material is deposited or
silently deleting water when saturated soil is eroded.

## Driver order

For each storm or interstorm interval:

1. Convert rainfall and PET to recharge and infiltration-excess runoff.
2. Advance `GroundwaterWetnessCoupler` over the interval.
3. Transfer surface runoff to `water__unit_flux_in`.
4. Run `PriorityFloodFlowRouter`.
5. Evaluate `ShallowLandslider` using the updated wetness and current PGA.
6. Run selected-source landslide transport when enabled.
7. Reconcile groundwater if soil, bedrock, or topography changed.
8. Rerun routing before SPACE or the next event.
9. Advance SPACE using a hydrologically and dimensionally consistent
   discharge/timestep treatment.

An earthquake uses the groundwater state at its event time. PGA is zero during
ordinary hydrologic intervals, but rainfall and drainage continue to modify
the state inherited by subsequent earthquakes.

## Current limitations

- Recharge is prescribed; rainfall-to-recharge conversion is not implemented.
- The Dupuit approximation neglects vertical groundwater gradients and assumes
  a shallow, laterally extensive unconfined aquifer with a small capillary
  fringe.
- The initial reconciliation policy treats deposited material as dry. A later
  runout model may provide water content for deposited material explicitly.
- No automatic multirate scheduler yet connects storm-scale hydrology to a
  coarser SPACE timestep.
- Hydraulic conductivity, drainable porosity, initial water table, recharge,
  and hydrologic boundaries require calibration and sensitivity analysis.

## Verification requirements

The automated tests check:

- exact conversion between saturated thickness and relative wetness;
- increasing wetness under positive recharge;
- live consumption of updated wetness by one ShallowLandslider instance;
- exclusion and numerical treatment of zero-soil nodes;
- runoff transfer without changing flow-routing calculations;
- refusal to advance against stale terrain;
- water expelled by erosion; and
- dry pore-space creation during deposition.

Future rainfall–recharge and event-driver tests must additionally close the
complete water balance and verify storm recession, earthquake–storm sequences,
and full-domain Nepal performance.
