# SkyMatch.jl

SkyMatch.jl implements the numerical forward model used to generate
SIDES-style millimeter-source catalogs. Given galaxy properties, model
parameters, and explicit random draws, it computes:

- quenched and starburst classifications and star-formation rates;
- gravitational magnification;
- dust SED quantities, monochromatic fluxes, and optional filter fluxes;
- CO, [CII], and [CI] line luminosities and fluxes.

The core entry point is `SkyMatch.forward_model(inputs, params, noise)`. It is
pure numerical Julia: file loading, random-number generation, backend
selection, compilation, and catalog output remain outside the function. The
same forward-model source therefore runs as ordinary Julia or compiles with
Reactant for CPU, NVIDIA GPU, and TPU execution.

## Setup

SkyMatch currently targets Julia 1.11. From the repository root:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The paths in `data/SIDES_from_original.par` must resolve to the model tables:

- `SED_finegrid_dict.h5`
- `LFIR_LIR_ratio.h5`
- `data/Psupmu_table_Bethermin17.txt`
- `data/Daddi15_SLED.txt`
- a SIDES CSV catalog

Input catalogs must contain `redshift`, `Mstar`, `Mhalo`, `ra`, and
`dec`. Filter-grid HDF5 files are required only when filter fluxes are
enabled.

## Forward-model API

```julia
using SkyMatch

configuration = SkyMatch.load_par_file("data/SIDES_from_original.par")
catalog = SkyMatch.load_sides_csv("data/SIDES_Bethermin2017_short2.csv")

inputs = SkyMatch.build_forward_inputs(catalog)
parameters = SkyMatch.build_forward_parameters(configuration; filters=false)
noise = SkyMatch.make_simulation_noise(length(inputs.redshift))

output = SkyMatch.forward_model(inputs, parameters.numeric, noise)
```

The explicit `noise` argument makes a run reproducible and lets all backends
consume identical random draws. The result is a named tuple of arrays; use
`SkyMatch.add_output_columns!` to attach those arrays to a catalog.

The Reactant path changes data placement and compilation, not the model:

```julia
using Reactant

reactant_inputs = Reactant.to_rarray(inputs)
reactant_parameters = Reactant.to_rarray(parameters.numeric)
reactant_noise = Reactant.to_rarray(noise)

compiled_forward = Reactant.@compile sync=true SkyMatch.forward_model(
    reactant_inputs,
    reactant_parameters,
    reactant_noise,
)
reactant_output = compiled_forward(
    reactant_inputs,
    reactant_parameters,
    reactant_noise,
)
```

## Production driver

`gen_pysides_from_original.jl` is a small Reactant driver around the forward
model. CPU is the default backend:

```bash
julia --project=. gen_pysides_from_original.jl
julia --project=. gen_pysides_from_original.jl --backend CUDA --large
julia --project=. gen_pysides_from_original.jl --backend TPU --large
```

It reports transfer, compilation, execution, materialization, and total time.
Add `--write-output` to write the configured FITS catalog, `--rows N` to
limit a run, or `--filters` to include configured filter fluxes. Run with
`--help` for all options.

## Sky maps and spectral cubes

`SkyMatch.make_maps(catalog, map_parameters, model_parameters)` bins catalog
columns named `S<filter>` (flux density in Jy) and writes FITS images. If a filter
column is absent, it computes it from the configured HDF5 filter grid and the
catalog's `redshift`, `Umean`, `LIR`, `mu`, `issb`, and `qflag` columns. Supply
`ra` and `dec` in degrees; otherwise coordinates are drawn once using
`map_parameters["field_size"]` and attached to the catalog. Seed Julia's RNG
before that call for reproducibility.

`SkyMatch.make_cube(catalog, model_parameters, cube_parameters)` adds continuum
and line emission in spectral channels. Frequency parameters (`freq_min`,
`freq_max`, `freq_resol`) are in Hz; model line rest frequencies are in GHz,
integrated catalog line fluxes are in Jy km/s, and numeric `Dlum` is in Mpc.
Channel centers follow `freq_min:freq_resol:freq_max`; a nonaligned upper bound
is excluded rather than changing the requested channel spacing.
Combined `save_full`/`save_all_lines` outputs are generated per enabled CII
recipe: enable `gen_cube_CII_Lagache` or `gen_cube_CII_de_Looze` for those files.

Internal images use `(y, x)` and cubes use `(frequency, y, x)`; FITS output uses
`(x, y)` and `(x, y, frequency)` to match its WCS headers. Beam smoothing uses
periodic FFT convolution, so sources near an edge can wrap across the field.
These map/cube operations run on the host after the numerical forward model.

A self-contained [synthetic-catalog demo](examples/README.md) produces sky maps
and Enzyme luminosity sensitivities without external catalogs or SED tables.

## Performance and reproducibility

The Julia-versus-Reactant harness is isolated under [`benchmark/`](benchmark/).
It records machine-readable timing, allocation, overhead, provenance, and
correctness data.

### Reproducing the paper figures

Run every command below from the SkyMatch.jl repository root. Raw benchmark
JSON is read from benchmark/results/, and generated paper figures, LaTeX table
fragments, summaries, and handoff notes are written to
benchmark/plots/generated/. Figure 1 is the implementation architecture
authored in the manuscript's LaTeX/TikZ source; this repository does not
generate a fig1 file.

Instantiate the isolated benchmark environment once:

```bash
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'
```

Generate every paper artifact from the canonical raw directory with:

```bash
julia --startup-file=no --history-file=no --project=benchmark \
  benchmark/plots/plot_results.jl \
  --results-dir benchmark/results \
  --output-dir benchmark/plots/generated
```

Paper item | Generated artifact | Description
---|---|---
Figure 1 | Not generated (LaTeX/TikZ) | SkyMatch implementation architecture
Figure 2 | benchmark/plots/generated/fig2_runtime_speedup.pdf | Warm runtime and speedup
Figure 3 | benchmark/plots/generated/fig3_overheads_amortization.pdf | Measured overheads and derived amortization
Figure 4 | benchmark/plots/generated/fig4_memory_scaling.pdf | GPU peak memory and host allocation traffic
Table II | benchmark/plots/generated/table2_largest_common.tex | 1,000,000-source comparison
Table III | benchmark/plots/generated/table3_correctness.tex | Julia/Reactant numerical agreement

The complete paper matrix consists of these eight commands. Every command writes
uniquely named sibling JSON files directly into the same canonical raw
directory:

```bash
julia --startup-file=no --history-file=no --project=benchmark benchmark/benchmark_reactant.jl --large --rows 1024 --backend CPU --results-dir benchmark/results
julia --startup-file=no --history-file=no --project=benchmark benchmark/benchmark_reactant.jl --large --rows 100000 --backend CPU --results-dir benchmark/results
julia --startup-file=no --history-file=no --project=benchmark benchmark/benchmark_reactant.jl --large --rows 1000000 --backend CPU --results-dir benchmark/results
julia --startup-file=no --history-file=no --project=benchmark benchmark/benchmark_reactant.jl --large --rows 5584998 --backend CPU --results-dir benchmark/results
julia --startup-file=no --history-file=no --project=benchmark benchmark/benchmark_reactant.jl --large --rows 1024 --backend CUDA --results-dir benchmark/results
julia --startup-file=no --history-file=no --project=benchmark benchmark/benchmark_reactant.jl --large --rows 100000 --backend CUDA --results-dir benchmark/results
julia --startup-file=no --history-file=no --project=benchmark benchmark/benchmark_reactant.jl --large --rows 1000000 --backend CUDA --results-dir benchmark/results
julia --startup-file=no --history-file=no --project=benchmark benchmark/benchmark_reactant.jl --large --rows 5584998 --backend CUDA --results-dir benchmark/results
```

Run each command once. Each invocation relies on the harness's internal
Chairmarks sampling for Julia and Reactant CPU or its synchronized 25-execution
XProf sampling for Reactant GPU; no external replication loop is needed.
Filters remain disabled, explicit noise uses seed 1234, ordinary Julia uses one
Julia thread, and Reactant CPU retains XLA's default CPU parallelization.

The 5,584,998-source GPU case may exhaust the usable memory of the selected RTX
5090. A recorded OOM is a valid reproduced outcome and is shown explicitly in
the generated artifacts; it is not an artifact-generation failure.

Expected results: Reactant CPU should cross over after the smallest catalog, and
the single GPU should be fastest at the larger successfully measured sizes.
These are qualitative checks only. Consult
benchmark/plots/generated/results_summary.json for the exact recorded values,
status of every point, correctness extrema, and break-even calculations.

## Tests

```bash
julia --startup-file=no --history-file=no --project=. test/runtests.jl
```

The regression suite checks Julia/Reactant agreement for the complete forward
model, input-table boundary cases, output assembly, and FITS writing.

## Source layout

- `src/reactant_pipeline.jl`: backend-independent numerical forward model.
- `src/pipeline_host.jl`: input preparation and table loading.
- `src/io.jl`: parameter, catalog, and FITS I/O.
- `data/`: tracked model inputs and local, ignored catalogs.
- `gen_pysides_from_original.jl`: Reactant execution and optional FITS output.
- `benchmark/`: benchmark runner and result utilities.
- `benchmark/plots/`: plotting scripts.
