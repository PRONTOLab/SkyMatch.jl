# Synthetic SkyMatch demo

This CPU example uses SkyMatch's star-formation model and the tracked
`data/SIDES_from_original.par` parameters. A fixed seed generates 2,048 synthetic
galaxies in a small sky field. Enzyme computes each galaxy's infrared luminosity
response to the `sigma_MS` scatter parameter, holding galaxies and random draws
fixed; a five-point finite difference checks the result.

The maps use an **illustrative flux scale**, `Sdemo[Jy] = LIR[Lsun] * 1e-15`.
This is not a physical observing band or a luminosity-distance conversion.
The example exercises the FITS map and beam-convolution code without requiring
untracked HDF5 tables, external catalogs, Python, or a GPU. It does not execute
the complete SED and line-emission forward model.

From the repository root with Julia 1.11:

```sh
julia --startup-file=no --project=examples -e 'using Pkg; Pkg.instantiate()'
julia --startup-file=no --project=examples examples/skymatch_demo.jl /tmp/skymatch-demo
```

The output directory contains a source map and beam map in FITS format, a PNG
figure, and JSON recording seeds, demo/SkyMatch source and parameter-file
checksums, environment checksums, package versions, derivative validation,
and flux conservation. Omitting the output argument writes to
`examples/generated/`. Keep generated output out of source commits. Julia's
first run includes compilation; this example does not claim benchmark speedups.

For exact reproduction, archive `examples/Project.toml` and the generated
`examples/Manifest.toml` with the outputs and the matching SkyMatch source
revision. The manifest is intentionally a local file ignored by the package
repository; its relative SkyMatch source path (`..`) assumes the repository
layout. The output's hashes identify the precise environment and source files
used, including uncommitted source changes that a package version cannot identify.

For a notebook, the code can be prepared separately from display:

```julia
include("examples/skymatch_demo.jl")
case = SkyMatchDemo.prepare_demo() # numerical work and derivative validation
result = SkyMatchDemo.run_demo(case; output_dir="/tmp/skymatch-demo")
result.figure
```

## SAIL integration

The candidate fits after the Enzyme heat example in
[`SAIL2026_demo.ipynb`](https://github.com/PRONTOLab/sail2026-demo/blob/main/SAIL2026_demo.ipynb).
Add a helper module there, include it from `demo_utils.jl`, and prepare the case
in `prepare_demo!()`. Keep the synthetic-data and toy-flux explanation beside
the figure. Pin a tested SkyMatch commit in `colab/colab_setup.jl` and save the
executed notebook outputs, following the existing notebook convention.

The SAIL setup inspected at commit `4bbf8e57c438b800ec5382a76014f24c0281f389`
uses Julia 1.11 on a Colab T4, Reactant's `impulse` branch with its matching
prebuilt `libReactantExtra` from release `v1`, Reactant_jll 0.0.424, Enzyme
0.13.216, CairoMakie 0.15.16, CUDA 6.2.2, and patched Comrade 0.11.33.
Test the new example inside that environment before changing the notebook;
this local example's environment does not establish Colab compatibility.
No benchmark runner or complete merge of `vim/plots2` is required: the small
LIR sensitivity kernel is adapted here, with the current `SkyMatch` name.
