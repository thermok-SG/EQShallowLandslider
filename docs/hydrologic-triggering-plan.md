# Hydrologic triggering implementation plan

## Objective

Allow `ShallowLandslider` to initiate shallow landslides when hydrologic change
reduces static stability, while retaining seismic and combined triggering. The
component should consume a hydrologic state through Landlab fields rather than
owning a particular rainfall or groundwater model.

The initial implementation should answer a narrow question reliably:

> Given a node-wise relative wetness state, which slopes are unstable, what is
> their probability of failure, and which failure regions should be sent to the
> existing geometry and runout machinery?

Rainfall generation and transient groundwater modelling can then be composed
around that stable interface. The larger objective is a time-evolving sequence
of ordinary storms and intermittent earthquakes in which every event inherits
the landscape produced by all preceding events.

## Time-evolving model architecture

This work is not just a choice between two trigger modes. It requires a
persistent landscape state, transient forcing, and derived fields that are
refreshed whenever the landscape changes.

### Persistent state

At minimum, the model must carry these fields through time:

- `bedrock__elevation`;
- `soil__depth`;
- `topographic__elevation`, maintained consistently as bedrock plus soil;
- the current hydrologic state, including `soil__relative_wetness` when field
  hydrology is active;
- any spatial soil-strength fields introduced later, such as cohesion or
  friction angle.

This persistent physical state, rather than an internal "already failed" mask,
is the primary memory of previous landslides. Erosion removes available soil,
deposition creates thicker potentially unstable material elsewhere, and
topographic change alters later routing and stability.

For a time-evolving simulation, runout and landscape updating are therefore
required rather than optional. A stability-only call remains useful for hazard
mapping or diagnostics, but it cannot represent geomorphic preconditioning.
Introduce an explicit model-level distinction such as:

```text
evolve_landscape = false  -> diagnose/select failures without persistent change
evolve_landscape = true   -> require runout and commit erosion/deposition
```

When `evolve_landscape=true`, configuration validation must require runout,
soil updating, an applicable runout-distance model, and flow-routing fields.
It should reject internally inconsistent combinations instead of silently
leaving the landscape unchanged.

### Transient forcing

- During an earthquake event, PGA fields contain that event's shaking while
  the current hydrologic state remains active.
- Outside an earthquake event, PGA is zero.
- During storms and interstorms, rainfall/recharge and groundwater evolve while
  stability is evaluated with zero PGA.

An earthquake can therefore occur on a wet or dry landscape, and subsequent
rainfall acts on coseismically modified soil, topography, and drainage.

### Derived state and invalidation

The current component caches slope and aspect at construction, and the current
runout code changes `soil__depth` without changing `topographic__elevation`.
Both behaviours must change for meaningful landscape evolution.

After erosion or deposition, the model must:

1. update `topographic__elevation = bedrock__elevation + soil__depth`;
2. refresh slope and aspect before the next stability calculation;
3. rerun flow routing when topography changes;
4. recompute contributing area and hydrologic wetness where required;
5. then evaluate the next storm or earthquake.

Use an explicit terrain-dirty/update contract rather than silently relying on
cached terrain attributes.

### Event loop

A model driver, rather than `ShallowLandslider` itself, should own time and
event scheduling:

```text
initialize persistent landscape and hydrology

for each storm, interstorm interval, or earthquake:
    update transient forcing
    advance hydrology to the event time
    refresh derived terrain/hydrology fields if the landscape is dirty
    evaluate stability and select failures
    run landslide transport
    update soil depth and topography
    mark routing and hydrologic derivatives dirty
    record event diagnostics and advance the clock
```

This separation permits observed event catalogues, stochastic rainfall, and
stochastic earthquake sequences to drive the same component stack.

Only failures actually selected and transported are committed to the landscape.
Unselected unstable candidates remain diagnostic possibilities and do not lose
soil. This makes the probabilistic selection rule a direct control on subsequent
landscape evolution and means its calibration must be recorded with each run.

## Existing mechanics

`ShallowLandslider` currently uses `submerged_soil_proportion`, denoted here by
`m`, in both factor-of-safety and critical-acceleration calculations. With
horizontal and vertical accelerations set to zero, its equations satisfy

```text
a_critical = g * sin(slope) * (factor_of_safety - 1)
```

Consequently, the existing instability test
`a_driving > a_critical` becomes `factor_of_safety < 1` without shaking. The
stability mechanics can therefore support static hydrologic failure after the
wetness input and earthquake-specific downstream assumptions are separated.

The current barriers are:

1. `submerged_soil_proportion` is a scalar captured at construction time.
2. Omitted PGA values create non-zero fallback shaking.
3. Both group-selection strategies use PGA.
4. Runout is only invoked through Newmark displacement and shaking duration.
5. There is no hydrologic analogue of the existing critical-acceleration ratio
   used by probabilistic group selection.

## Hydrologic state contract

### Canonical field

When field-driven hydrology is enabled, require a node field:

```text
soil__relative_wetness [-], constrained to [0, 1]
```

Define relative wetness as the saturated thickness divided by the potentially
unstable soil thickness. This matches the role of `m` in a SINMAP-style
infinite-slope model. It is deliberately distinct from root-zone volumetric
water saturation.

`ShallowLandslider` will read the field at every `run_one_step`, so another
component can update it between stability calculations. A configuration such
as `wetness_source="constant"` or `wetness_source="field"` should control the
contract:

- `"constant"` preserves the existing component interface and uses the scalar
  `submerged_soil_proportion` as a fixed background pore-pressure condition.
- `"field"` requires `soil__relative_wetness`; absence of the field is an
  immediate, descriptive error.

This remains one `ShallowLandslider` component. The flag changes where it gets
wetness, not which stability model it uses. A future major interface may make a
node field mandatory in every mode, but that is outside the initial work.

Validation should be strict: reject non-finite core-node values and values
outside `[0, 1]`; do not silently clip state supplied by another model.

### Evaluation state

Expose the current physical state:

- `landslide__is_unstable`: the current physical state.

Every evaluation treats all currently unstable nodes as eligible candidates.
The core component will not infer whether two calls are independent events or
adjacent timesteps in one storm. If a continuous simulation needs to trigger
only on stable-to-unstable transitions, its driver can compare successive
`landslide__is_unstable` fields. Normally, erosion and deposition should change
the physical state after a selected failure; if landscape updating is disabled,
the same unstable candidate may legitimately be selected again on a later
evaluation.

Keep the physical instability calculation unified: seismic, hydrologic, and
combined forcing differ only in the acceleration and wetness fields supplied.
Do not create separate copies of the factor-of-safety equation.

## Reuse assessment

| Existing component or subsystem | Reuse | Rationale |
| --- | --- | --- |
| `ShallowLandslider` stability equations | Reuse after array-enabling `m` | The zero-PGA limit already represents static failure. |
| Region labelling, hole filling, aspect grouping, width splitting, properties | Reuse directly | These operate on an instability mask and are trigger-agnostic. |
| Existing probabilistic/PGA-weighted selection | Retain its group-selection structure for seismic use; generalise the probability input | PGA should not be used to assign hydrologic probability, but the group aggregation and weighted draw can be reused. |
| `ShallowLandslideRunout` | Reuse after decoupling its source-node selection from Newmark displacement | Routing is independent of the cause of initiation. |
| Landlab `LandslideProbability` | Adapt its relative-wetness concept and field requirements; do not instantiate it internally | It combines steady-state SINMAP wetness with Monte Carlo susceptibility and uses global random sampling. It is not a transient storm model. |
| `PriorityFloodFlowRouter` | Reuse drainage area and existing routing setup | It is already part of the CLI workflow. Specific contributing area still needs an explicit conversion or field. |
| `PrecipitationDistribution` | Reuse in example drivers | It supplies storm/interstorm timing and spatially uniform `rainfall__flux`; it should not be a dependency of `ShallowLandslider`. |
| `SpatialPrecipitationDistribution` | Reuse optionally in example drivers | It supplies node-wise `rainfall__flux`, but its default calibration is climate-specific. |
| `GroundwaterDupuitPercolator` | Reuse in a later transient integration | If its aquifer base represents the failure-plane/bedrock surface, `aquifer__thickness / soil__depth` can supply relative wetness. |
| Landlab `SoilMoisture` | Do not map directly in the first implementation | Its cell-based root-zone volumetric saturation is not the saturated-thickness ratio required by the stability equation. Leakage may later provide recharge to groundwater. |

## Proposed components and responsibilities

### 1. `ShallowLandslider`

Responsibilities:

- Read current relative wetness from the canonical field or scalar fallback.
- Calculate static and acceleration-dependent stability.
- Identify current unstable regions.
- Apply trigger-appropriate selection.
- Pass selected failures to runout independently of Newmark displacement.

An `"all"` selection method may be useful for analytical tests, deterministic
hazard envelopes, and debugging; it means that every candidate landslide is
selected. It is not the primary hydrologic occurrence model.

For hydrologic probability, derive a critical relative wetness, `m_c`, by
solving the existing ShallowLandslider factor-of-safety equation for `m` at
`FS = 1`:

```text
m_c = (cohesion
       + soil_unit_weight * soil_depth * sin(slope)
         * (tan(friction_angle) / tan(slope) - 1))
      / (water_unit_weight * soil_depth * tan(friction_angle))
```

This is the hydrologic analogue of critical acceleration `a_c`:

- `m_c <= 0`: unstable even when dry.
- `0 < m_c <= 1`: hydrologically triggerable, with `m / m_c` expressing how
  close current wetness is to the threshold.
- `m_c > 1`: cannot be triggered by saturation alone under the current
  parameter values.

The existing group-probability machinery can be generalised to accept a
probability field produced from hydrologic state instead of always calculating
probability from PGA. Two probability levels should be kept distinct:

1. A provisional, empirical mapping from `m / m_c` to a selection weight,
   analogous to the current PGA/`a_c` mapping. This supports the first coupled
   implementation but requires calibration and should be called a weight, not
   a physically derived probability.
2. A later Monte Carlo probability of `FS <= 1`, calculated using uncertain
   cohesion, friction, soil depth, and wetness/recharge while retaining the
   ShallowLandslider equation. This is the scientifically preferred probability
   model.

Landlab's `LandslideProbability` is useful as a design reference for the second
level: parameter distributions, iteration structure, and probability-of-
failure outputs can be adapted. It should not be called directly because it
uses the SINMAP factor-of-safety equation rather than the ShallowLandslider
equation that this project intends to preserve.

### 2. `SinmapWetness`

Implement a small, vectorised Landlab component or adapter that writes
`soil__relative_wetness` using

```text
m = min((recharge / transmissivity) *
        (specific_contributing_area / sin(slope)), 1)
```

Inputs:

- `groundwater__recharge` at node, in length/time, or a scalar/array argument.
- `soil__transmissivity` at node, in length squared/time; alternatively compute
  it explicitly from saturated hydraulic conductivity times soil depth.
- `topographic__specific_contributing_area` at node, in length.
- topographic slope, computed from elevation or accepted as an existing field.

Output:

- `soil__relative_wetness` at node.

This component should borrow the tested relationship and terminology from
Landlab's `LandslideProbability`, but not copy its Monte Carlo loop. It provides
a deterministic steady-state response to each supplied recharge value. A
storm sequence can update recharge and rerun it, with the limitation that this
is a sequence of steady states rather than transient infiltration.

Specific contributing area must not be silently equated to drainage area. For
a raster-only convenience option, document and test the approximation
`drainage_area / cell_width`; otherwise require the correctly calculated field.

### 3. Hydrology adapters and drivers

Keep orchestration outside `ShallowLandslider`:

```text
rainfall generator
    -> infiltration/recharge calculation
    -> wetness or groundwater component
    -> ShallowLandslider.run_one_step()
    -> optional runout
```

The first example driver should use prescribed recharge and `SinmapWetness`.
A second integration can use `GroundwaterDupuitPercolator`, setting the aquifer
base consistently with the soil failure layer and converting its saturated
thickness to relative wetness. Connecting Landlab `SoilMoisture` leakage to
groundwater recharge should be treated as a separate research/validation task.

## Runout and displacement

Newmark displacement remains a seismic diagnostic. Hydrologic failure has no
shaking duration and should not synthesize one.

Refactor runout invocation to accept initiation source nodes from either:

- seismic mode: selected nodes exceeding the configured displacement threshold;
- static/hydrologic mode: selected unstable nodes;
- combined mode: an explicitly configured union or displacement rule.

The source-volume and travel-distance rules for hydrologic failures must be
documented. The existing runout tracer requires a Newmark-derived maximum
distance, so merely bypassing the displacement threshold is insufficient.
Hydrologic runout needs an independent distance provider, initially one of:

- an explicitly supplied `landslide__runout_distance` field;
- a configurable empirical distance/angle-of-reach rule; or
- a callable mobility model based on source volume and terrain.

The first implementation should support an explicit field or callable, keeping
the transport code reusable without pretending that rainfall has a shaking
duration. After transport, runout must update both soil depth and topographic
elevation consistently.

## Preconditions represented by the evolving landscape

The initial model will represent event-to-event preconditioning through:

- loss of soil at failed source nodes;
- accumulation and loading at deposition nodes;
- changes in slope, aspect, contributing area, and flow paths;
- changes in soil transmissivity or storage caused by changing soil depth;
- the hydrologic state at the time of an earthquake or rainfall event.

Potential later processes include regolith production, weathering, vegetation
and root-cohesion recovery, earthquake damage to strength, and compaction or
hydraulic-property changes in deposits. These require explicit evolution laws
and should not be hidden inside the trigger calculation.

## Equation review before calibration

Before comparing probabilities or calibrating against inventories, write the
current ShallowLandslider factor-of-safety equation and the SINMAP/Landlab
equation in the documentation with consistent definitions of soil depth,
normal stress, pore pressure, density, and slope angle. Their water-pressure
terms are not currently identical.

The present ShallowLandslider equation is authoritative for this implementation.
Enabling hydrology must not replace it with the SINMAP equation or silently
alter established seismic results. The SINMAP comparison remains useful for
understanding assumptions and adapting hydrologic inputs, but changing the
stability equation is explicitly out of scope.

## Delivery stages

### Stage 1: dynamic wetness and static initiation

**Status: implemented on `feat/hydrologic-triggering`.** The live field
contract, zero omitted PGA, unchanged array-enabled stability calculations,
critical relative wetness diagnostic, validation, analytical tests, and user
documentation are in place. Generalised hydrologic probability/selection is
deliberately deferred to the later uncertainty stage rather than being given an
uncalibrated probability interpretation in this foundation change.

- Add and document the optional node wetness field.
- Generalise both stability functions from scalar to scalar-or-node array.
- Read wetness on every step.
- Require the field when `wetness_source="field"` and preserve scalar wetness
  when `wetness_source="constant"`.
- Make omitted PGA zero without a deprecation path.
- Add and test critical relative wetness `m_c`.

Exit criterion: on a synthetic slope with zero PGA, increasing wetness across
the analytical threshold creates the expected unstable region without
reconstructing the component.

### Stage 2: landscape-state and runout consistency

- Add `evolve_landscape` validation that makes runout mandatory for temporal
  simulations.
- Select runout sources according to the initiation mechanism and selection
  result.
- Allow hydrologic failures to reach runout with displacement disabled.
- Add an explicit hydrologic runout-distance field/callable.
- Update topographic elevation consistently with erosion and deposition.
- Refresh slope and aspect after terrain change.
- Establish a terrain-dirty contract for rerouting and hydrology.
- Verify soil removal prevents or controls repeated failure as intended.

Exit criterion: a zero-PGA wetness event initiates a failure and produces
runout using existing hill-flow routing fields; the next stability evaluation
uses the modified soil and topography.

### Stage 3: deterministic SINMAP wetness adapter

- Implement `SinmapWetness` with scalar and spatial recharge.
- Reuse existing flow-routing output where physically valid.
- Add an example prescribed-storm/recharge driver.

Exit criterion: recharge increases wetness monotonically, wetness is capped at
one, and an analytical small-grid case produces the expected failure threshold.

### Stage 4: transient groundwater integration

- Compose rainfall/recharge with `GroundwaterDupuitPercolator`.
- Define and validate the relationship between its aquifer base, saturated
  thickness, soil depth, and the landslide failure plane.
- Exercise both storm and drainage/recovery periods.

Exit criterion: a storm hydrograph raises and lowers node wetness while the
unstable and selected regions respond to the current state at each evaluation.

### Stage 5: event-sequence driver

- Implement a model clock and explicit storm/interstorm/earthquake events.
- Reset PGA to zero outside earthquake intervals.
- Reroute and update hydrology after landscape change.
- Record event type, time, forcing, failures, erosion, deposition, and state
  snapshots sufficient to attribute preconditioning.
- Demonstrate an earthquake followed by rainfall, and rainfall followed by a
  second earthquake, on one persistent grid.

Exit criterion: the second event produces different stability and failure
patterns because it inherits the first event's landscape changes.

### Stage 6: uncertainty and probability

- Decide whether to provide Monte Carlo parameter sampling as a separate
  wrapper or accept an externally supplied failure-probability field.
- Adapt the uncertainty structure of `LandslideProbability` while evaluating
  the ShallowLandslider stability equation.
- Use local `numpy.random.Generator` instances rather than global RNG state.
- Keep physical instability probability distinct from post-processing that
  subsamples geometrically connected candidate regions.

### Stage 7: CLI and ensemble support

- Add hydrology configuration only after the component contracts settle.
- Support fixed wetness, prescribed recharge/SINMAP, and external field modes.
- Record forcing mode, wetness source, units, and hydrology parameters in model
  outputs.
- Define chunked-mode constraints; catchment-scale contributing area and
  groundwater flow cannot generally be recomputed independently per tile.

## Test plan

### Analytical unit tests

- Scalar wetness produces the same results as a spatially uniform node field.
- Wetness zero and one match hand-calculated factor of safety and critical
  acceleration.
- At zero PGA, `a_critical = g sin(slope) (FS - 1)`.
- Invalid wetness values and incorrect array sizes raise clear errors.
- Wetness is read live rather than cached at construction.

### Evaluation and selection tests

- Every call exposes the currently unstable mask without hidden event history.
- Changing the wetness field between calls immediately changes instability.
- Critical wetness reproduces the `FS = 1` threshold analytically.
- Hydrologic weights are monotonic in `m / m_c` for triggerable slopes.
- A supplied probability/weight field can drive the reusable group selector.
- Deterministic selection retains all candidate labels.

### Regression tests

- Existing scalar wetness and explicitly supplied PGA reproduce current
  seismic stability arrays.
- Existing seismic selection remains available and unchanged.
- Region geometry and splitting tests remain trigger-agnostic.

### Integration tests

- Recharge -> SINMAP wetness -> zero-PGA failure on a small raster.
- Combined high wetness and shaking can fail a slope stable under either forcing
  alone.
- Hydrologic initiation reaches runout without Newmark displacement.
- Erosion/deposition keeps elevation, bedrock, and soil depth consistent.
- Slope, flow routing, contributing area, and wetness are refreshed after a
  landslide changes topography.
- In an earthquake -> rainfall -> earthquake sequence, later events inherit
  earlier soil and topographic changes while PGA is zero between earthquakes.
- Full-grid and supported chunked paths agree where hydrologic inputs are
  precomputed globally.

## Agreed design decisions

1. Each call evaluates all currently unstable nodes; the component does not
   impose stable-to-unstable event memory.
2. Omitted PGA becomes zero without a deprecation path.
3. `"all"` means selecting every candidate and is retained only as a useful
   deterministic/diagnostic option. Hydrologic probabilistic selection will use
   critical wetness and a generalised probability-input interface.
4. `wetness_source="constant"` retains the current scalar wetness interface.
   `wetness_source="field"` requires `soil__relative_wetness` to exist.
5. The existing ShallowLandslider stability equation remains authoritative;
   SINMAP supplies reusable hydrologic and uncertainty ideas, not a replacement
   stability equation.

## Remaining probability decisions

1. Choose the provisional mapping from `m / m_c` to group selection weight.
2. Decide which parameter distributions the later Monte Carlo model supports
   and whether probability is evaluated per node or per candidate group.
3. Define combined seismic-hydrologic probability without double-counting the
   wetness effect already present in critical acceleration.
