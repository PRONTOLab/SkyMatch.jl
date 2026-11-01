# BayesMMfwd.jl

BayesMMfwd.jl implements the numerical forward model used to generate
SIDES-style millimeter-source catalogs. Given galaxy properties, model
parameters, and explicit random draws, it computes:

- quenched and starburst classifications and star-formation rates;
- gravitational magnification;
- dust SED quantities, monochromatic fluxes, and optional filter fluxes;
- CO, [CII], and [CI] line luminosities and fluxes.

The core entry point is `BayesMMfwd.forward_model(inputs, params, noise)`. It is
pure numerical Julia: file loading, random-number generation, backend
selection, compilation, and catalog output remain outside the function. The
same forward-model source therefore runs as ordinary Julia or compiles with
Reactant for CPU, NVIDIA GPU, and TPU execution.

## Setup

BayesMMfwd currently targets Julia 1.11. From the repository root:

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
using BayesMMfwd

configuration = BayesMMfwd.load_par_file("data/SIDES_from_original.par")
catalog = BayesMMfwd.load_sides_csv("data/SIDES_Bethermin2017_short2.csv")

inputs = BayesMMfwd.build_forward_inputs(catalog)
parameters = BayesMMfwd.build_forward_parameters(configuration; filters=false)
noise = BayesMMfwd.make_simulation_noise(length(inputs.redshift))

output = BayesMMfwd.forward_model(inputs, parameters.numeric, noise)
```

The explicit `noise` argument makes a run reproducible and lets all backends
consume identical random draws. The result is a named tuple of arrays; use
`BayesMMfwd.add_output_columns!` to attach those arrays to a catalog.

The Reactant path changes data placement and compilation, not the model:

```julia
using Reactant

reactant_inputs = Reactant.to_rarray(inputs)
reactant_parameters = Reactant.to_rarray(parameters.numeric)
reactant_noise = Reactant.to_rarray(noise)

compiled_forward = Reactant.@compile sync=true BayesMMfwd.forward_model(
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

## Performance and reproducibility

The Julia-versus-Reactant harness is isolated under [`benchmark/`](benchmark/).
It records machine-readable timing, allocation, overhead, provenance, and
correctness data.

### Reproducing the paper figures

Run every command below from the BayesMMfwd.jl repository root. Raw benchmark
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
Figure 1 | Not generated (LaTeX/TikZ) | BayesMMfwd implementation architecture
Figure 2 | benchmark/plots/generated/fig2_runtime_speedup.pdf | Warm runtime and speedup
Figure 3 | benchmark/plots/generated/fig3_overheads_amortization.pdf | Measured overheads and derived amortization
Figure 4 | benchmark/plots/generated/fig4_memory_scaling.pdf | GPU peak memory and host allocation traffic
Figure 5 | benchmark/plots/generated/fig5_implementation_runtime.pdf | pySIDES-to-GPU model-runtime scaling
Figure 6 | benchmark/plots/generated/fig6_execution_ablation.pdf | 1,000,000-source execution-overhead ablation
Figure 7 | benchmark/plots/generated/fig7_sfr_sensitivity_runtime.pdf | Enzyme SFR-sensitivity runtime
Table II | benchmark/plots/generated/table2_largest_common.tex | 1,000,000-source comparison
Table III | benchmark/plots/generated/table3_correctness.tex | Julia/Reactant numerical agreement
Table IV | benchmark/plots/generated/table4_sfr_sensitivity.tex | a0-to-SFR sensitivity agreement

The existing Julia/Reactant forward matrix consists of these eight commands. Every command writes
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
For Figure 5, use the pinned public pySIDES release and its isolated Python
environment:

```bash
git clone https://gitlab.lam.fr/mbethermin/sides-public-release.git sides-public-release
git -C sides-public-release checkout b447ca818e62fd54ca0da797b737e7485af79b03
python3 -m venv benchmark/.venv-pysides
benchmark/.venv-pysides/bin/pip install -r benchmark/pysides-requirements.txt
for n in 1024 100000 1000000 5584998; do
  benchmark/.venv-pysides/bin/python benchmark/benchmark_pysides.py \
    --rows "$n" --results-dir benchmark/results --timeout-seconds 600
done
```

pySIDES uses its released default `multiprocessing.Pool(cpu_count())` policy.
Figure 5 reports each implementation's native model-execution scope: pySIDES
includes its complete pandas host catalog, while the existing Reactant values
are warm resident executions. It is not an end-to-end I/O comparison.

Generate the SFR sensitivity data for Figure 7 and Table IV with:

```bash
for backend in CPU CUDA; do
  for n in 1024 100000 1000000 5584998; do
    julia --startup-file=no --history-file=no --project=benchmark \
      benchmark/benchmark_sfr_sensitivity.jl --rows "$n" \
      --backend "$backend" --results-dir benchmark/results
  done
done
```

This differentiates `mean(log1p(SFR / sfr_scale))` with respect to `a0`
while holding the catalog and seed-1234 noise fixed. Table IV checks Enzyme
against a five-point central finite difference at 1,000,000 sources.

The default plotting command requires the forward matrix, pySIDES results,
SFR sensitivity results, and the 1,000,000-source Julia compilation measurements.
Older forward-only results are insufficient for Figures 5–7 and Table IV.

For the additional per-galaxy infrared-luminosity sensitivity plot, run:

```bash
julia --startup-file=no --history-file=no --project=benchmark benchmark/test_lir_sensitivity.jl
for backend in CPU CUDA; do
  for n in 1024 100000 1000000 5584998; do
    julia --startup-file=no --history-file=no --project=benchmark \
      benchmark/benchmark_lir_sensitivity.jl --rows "$n" \
      --backend "$backend" --results-dir benchmark/results
  done
done
julia --startup-file=no --history-file=no --project=benchmark \
  benchmark/plots/plot_results.jl --results-dir benchmark/results \
  --output-dir benchmark/plots/generated --only lir-sensitivity
```

This uses forward-mode Enzyme to calculate `dLIR_i/dsigma_MS`, holding galaxy
classifications and seed-1234 random draws fixed. It measures the SFR-to-LIR
calculation, not the complete forward model. CPU derivatives are checked against
five-point finite differences at two step sizes; Reactant derivatives are checked
against Julia. The default plotting command also generates these artifacts when
LIR results are present. Preserve the raw result JSON and environment manifests
alongside archived paper artifacts to retain the exact measured runs; they are
local outputs and are not included in the source repository.

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
These are qualitative checks only. Consult the generated
`results_summary.json` for the original forward matrix, the
`pysides_forward_*_CPUbenchmarks.json` files for Figure 5, and the
`bayesmmfwd_sfr_sensitivity_*benchmarks.json` files for Figure 7. Their
correctness siblings provide the Table IV values.

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
