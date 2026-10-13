# Model run inventory

Last audited: 7 October 2026

This document records the model combinations for which local output evidence exists in:

- `EQShallowLandslider_HPC/analysis_output` (primary evidence)
- `EQShallowLandslider_HPC/runs` (used where complete run artifacts remain locally)
- `EQShallowLandslider` (checked for outputs and configuration context)

The main repository currently contains no generated run products outside its input data. Its YAML files are therefore **not** counted as completed runs. A member `config.yaml` or an ensemble manifest alone is also not counted as a completed run.

## Status key

- **Analysed** — the run appears in `region_ensemble.csv` and has analysis plots.
- **Plot only** — a run-specific analysis plot exists, but the run is absent from the combined `region_ensemble.csv`.
- **Artifacts** — a run directory contains `manifest.json`, `summary.json`, `regions.csv`/Parquet and raster output.
- **Configured only** — a manifest/member configuration exists, but no local output proves completion.

Unless stated otherwise, runs use probabilistic selection and random seed 5000. In the compact tables, `10`, `25`, `50`, `75`, and `100` mean submerged-soil proportions 0.10, 0.25, 0.50, 0.75, and 1.00.

## At-a-glance summary

| Study / output group | Locally evidenced run instances | Model coverage | Evidence |
|---|---:|---|---|
| Japan ensemble | 45 | Full 3 × 3 × 5 soil-parameter grid; elevation–linear soil depth; uniform PGA | 40 analysed + 5 plot only |
| Japan single runs | 2 | Two soil-depth models at the central soil parameters | Analysed |
| Kaikōura ensemble | 38 | 23 members of the base soil grid, plus 15 PGA-sweep runs | Analysed |
| Kaikōura single run | 1 | Elevation–linear central soil parameters | Analysed |
| Kaikōura ShakeMap runs | 2 | Elevation–linear and curvature–linear at the selected soil parameters | Analysed |
| Nepal single run | 1 | Mean-elevation/curvature–local-linear model at central soil parameters | Analysed |
| Synthetic Southern Alps: SPACE | 2 | Two raster-soil-depth parameter/PGA combinations | Artifacts |
| Synthetic Southern Alps: weathering Taylor | 2 | Two raster-soil-depth parameter/PGA combinations | Artifacts |

There are **89 run-specific analysis summaries** (45 + 2 + 38 + 1 + 2 + 1) and **4 additional synthetic run artifact sets**. Kaikōura contains repeated executions of two full parameter combinations during the PGA experiment, so run-instance totals should not be interpreted as counts of unique parameter combinations.

## Japan

### Elevation–linear ensemble

Fixed settings recoverable from the member configurations are soil-depth distribution `elevation`, relationship `linear`, uniform horizontal PGA 0.5 g, vertical PGA 0.2 g, probabilistic selection, and seed 5000.

All 45 combinations of effective cohesion, internal-friction angle and submerged-soil proportion produced run-specific analysis plots.

| Effective cohesion (Pa) | φ = 25° | φ = 30° | φ = 35° |
|---:|---|---|---|
| 10,000 | 10, 25, 50, 75, 100 | 10, 25, 50, 75, 100 | 10, 25, 50, 75, 100 |
| 15,000 | 10, 25, 50, 75, 100 | 10, 25, 50, 75, 100 | 10, 25, 50, 75, 100 |
| 20,000 | 10*, 25*, 50, 75, 100 | 10*, 25*, 50, 75, 100 | 10*, 25, 50, 75, 100 |

`*` Plot only. These five runs have both summary and map plots but are missing from `japan_ensemble/region_ensemble.csv`:

- c20,000 / φ25° / submerged proportion 0.10
- c20,000 / φ25° / submerged proportion 0.25
- c20,000 / φ30° / submerged proportion 0.10
- c20,000 / φ30° / submerged proportion 0.25
- c20,000 / φ35° / submerged proportion 0.10

Run dates encoded in the output IDs: 7 August 2026.

### Single comparison runs

| Soil-depth distribution | Relationship | Cohesion (Pa) | φ | Submerged proportion | Evidence |
|---|---|---:|---:|---:|---|
| `mean_elev_curv` | `linear_std_local` | 15,000 | 30° | 0.50 | Analysed (7 August 2026) |
| `elevation` | `linear` | 15,000 | 30° | 0.50 | Analysed (9 August 2026) |

## Kaikōura

### Base elevation–linear ensemble

The base ensemble uses elevation–linear soil depth, uniform horizontal PGA 0.5 g and vertical PGA 0.2 g. The manifest defines a 45-member grid, but analysis output confirms only the following 23 members:

| Effective cohesion (Pa) | φ = 25° | φ = 30° | φ = 35° |
|---:|---|---|---|
| 10,000 | 10, 25, 50, 75, 100 | 10, 25, 50 | — |
| 15,000 | 10, 25, 50, 75, 100 | 10, 25, 50, 75, 100 | 10, 25, 50, 75, 100 |
| 20,000 | — | — | — |

Run dates encoded in these output IDs: 25–29 September 2026.

### Uniform-PGA sensitivity runs

These use elevation–linear soil depth with cohesion 10,000 Pa and φ25°. The analysis metadata explicitly records horizontal PGA. Vertical PGA is not recorded in the analysis tables.

| Submerged proportion | Completed horizontal PGA values (g) | Run instances | Note |
|---:|---|---:|---|
| 0.10 | 0.1, 0.2, 0.3, 0.5, 0.7, 1.0 | 6 | The 0.5 g combination repeats a base-ensemble combination. |
| 0.50 | 0.1, 0.2, 0.3, 0.5, 0.7, 1.0 | 6 | The 0.5 g combination repeats a base-ensemble combination. |
| 1.00 | 0.1, 0.2, 0.3 | 3 | A separate base-ensemble output exists at 0.5 g. |

Run dates encoded in these output IDs: 30 September–3 October 2026. The base ensemble and PGA experiment together contain 38 analysed run instances representing 36 unique fully specified combinations.

### Single and ShakeMap runs

| Output group | Soil-depth distribution | Relationship | Cohesion (Pa) | φ | Submerged proportion | PGA model | Evidence |
|---|---|---|---:|---:|---:|---|---|
| `kaikoura_single` | `elevation` | `linear` | 15,000 | 30° | 0.50 | Uniform; exact amplitude not stored in analysis CSV | Analysed (25 September 2026) |
| `kaikoura_shakemap_curvature_linear` | `curvature` | `linear` | 10,000 | 30° | 0.10 | Raster/ShakeMap | Analysed (6 October 2026) |
| `kaikoura_shakemap_elevation_linear_corrected` | `elevation` | `linear` | 10,000 | 30° | 0.10 | Raster/ShakeMap | Analysed (6 October 2026) |

The current ShakeMap configuration states a vertical-to-horizontal PGA ratio of 0.4, but this setting is not present in the analysis CSV and should not automatically be assumed for older outputs. The local `shakemap_best_fit/run.log` ends during processing and is not itself evidence of completion; the two later analysis-output sets are the completion evidence used here.

## Nepal (Gorkha)

One completed single model is visible in the analysis output:

| Soil-depth distribution | Relationship | Cohesion (Pa) | φ | Submerged proportion | Evidence |
|---|---|---:|---:|---:|---|
| `mean_elev_curv` | `linear_std_local` | 15,000 | 30° | 0.50 | Analysed (6 August 2026) |

A 45-member Nepal elevation–linear ensemble is configured with the same 3 × 3 × 5 soil grid as Japan. Member configuration directories exist, but there is no Nepal ensemble analysis output or retained run artifact locally, so all 45 are classified as **configured only**.

## Synthetic Southern Alps

Both synthetic landscapes have 270-member manifests spanning:

- cohesion: 10,000, 15,000, 20,000 Pa
- internal-friction angle: 25°, 30°, 35°
- submerged-soil proportion: 0.10, 0.25, 0.50, 0.75, 1.00
- horizontal PGA: 0.1, 0.2, 0.3, 0.5, 0.7, 1.0 g

Only two members per landscape have complete local artifact sets. Soil depth is read from the corresponding raster and the configured relationship is linear. Vertical PGA is 0.4 times horizontal PGA.

| Landscape / soil-depth raster | Cohesion (Pa) | φ | Submerged proportion | Horizontal PGA (g) | Vertical PGA (g) | Evidence |
|---|---:|---:|---:|---:|---:|---|
| SPACE | 15,000 | 30° | 0.50 | 0.5 | 0.2 | Artifacts (29 August 2026) |
| SPACE | 10,000 | 25° | 1.00 | 1.0 | 0.4 | Artifacts (29 August 2026) |
| Weathering Taylor | 15,000 | 30° | 0.50 | 0.5 | 0.2 | Artifacts (29 August 2026) |
| Weathering Taylor | 10,000 | 25° | 1.00 | 1.0 | 0.4 | Artifacts (29 August 2026) |

The remaining 268 members in each synthetic manifest are **configured only** on the evidence currently available locally.

## Configured grids without completion evidence

| Study | Configured members | Locally evidenced from that grid | Missing local completion evidence |
|---|---:|---:|---:|
| Japan elevation–linear ensemble | 45 | 45 plot outputs (40 aggregated) | 0, although 5 need aggregation/recovery |
| Kaikōura base elevation–linear ensemble | 45 | 23 | 22 |
| Nepal elevation–linear ensemble | 45 | 0 | 45 |
| Synthetic Southern Alps SPACE | 270 | 2 | 268 |
| Synthetic Southern Alps weathering Taylor | 270 | 2 | 268 |

## Audit notes

- Run IDs encode cohesion (`c`), friction angle (`phi`), submerged-soil percentage (`sub`), soil-depth distribution/relationship, selection method and seed. PGA is not encoded in the run ID; the Kaikōura PGA values above come from `parameter_sensitivity_pga_horizontal_max.csv`, and the synthetic values come from each run manifest.
- Multiple member directories with the same member index exist for some ensembles. These appear to be regenerated configurations and are not treated as multiple runs without output evidence.
- A plot proves that analysis was produced for a run but does not prove that its raw model artifacts are still present locally.
- This inventory intentionally reports what can be supported by the current local files; it does not infer that missing HPC outputs were never run.
