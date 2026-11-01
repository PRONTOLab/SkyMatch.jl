module SkyMatchDemo

using SkyMatch
using CairoMakie
using DataFrames: DataFrame
using Enzyme: Enzyme
using FITSIO: FITS
using JSON: JSON
using Random: MersenneTwister, rand
using SHA: sha256

export prepare_demo, run_demo, lir_per_galaxy, lir_sensitivity

const PARAMETER_FILE = joinpath(pkgdir(SkyMatch), "data", "SIDES_from_original.par")
const DEFAULT_OUTPUT = joinpath(@__DIR__, "generated")

"""Intrinsic LIR in solar luminosities; only the scatter sigma_MS is active."""
function lir_per_galaxy(theta, redshift, mstar, params, noise, sfr2lir)
    varied = merge(params, (sigma_MS=sum(theta),))
    return sfr2lir .* SkyMatch.generate_sfr(redshift, mstar, varied, noise).SFR
end

"""One forward-mode Enzyme pass produces dLIR/dsigma_MS for every galaxy."""
function lir_sensitivity(theta, redshift, mstar, params, noise, sfr2lir)
    return Enzyme.autodiff(
        Enzyme.Forward, lir_per_galaxy, Enzyme.Duplicated,
        Enzyme.Duplicated(theta, one.(theta)),
        Enzyme.Const(redshift), Enzyme.Const(mstar), Enzyme.Const(params),
        Enzyme.Const(noise), Enzyme.Const(sfr2lir),
    )[1]
end

"""
    prepare_demo(; n=2048, seed=1234)

Create a deterministic SYNTHETIC catalog in a small rectangular sky field,
evaluate SkyMatch's SFR-to-LIR kernel, and check its Enzyme derivative against
a five-point finite difference. No external catalogs or SED tables are needed.
"""
function prepare_demo(; n::Int=2048, seed::Int=1234)
    n >= 2 || throw(ArgumentError("n must be at least 2"))
    raw = SkyMatch.load_par_file(PARAMETER_FILE)
    params = SkyMatch.parse_sfr_numeric_params(raw)
    rng = MersenneTwister(seed)
    catalog = DataFrame(
        ra=150.0 .+ 0.08 .* rand(rng, n),
        dec=2.0 .+ 0.05 .* rand(rng, n),
        redshift=0.1 .+ 3.9 .* rand(rng, n),
        Mstar=10.0 .^ (9.0 .+ 2.5 .* rand(rng, n)),
    )
    draws = SkyMatch.make_simulation_noise(n; seed=seed + 1)
    noise = (; draws.q_uniform, draws.sb_uniform, draws.sfr_uniform)
    args = ([params.sigma_MS], catalog.redshift, catalog.Mstar, params, noise, Float64(raw["SFR2LIR"]))
    luminosity = lir_per_galaxy(args...)
    derivative = lir_sensitivity(args...)
    h = 1.0e-4
    params.sigma_MS > 2h || error("sigma_MS is too small for the finite-difference stencil")
    evaluate(delta) = lir_per_galaxy([params.sigma_MS + delta], args[2:end]...)
    fd = (8 .* (evaluate(h) .- evaluate(-h)) .- (evaluate(2h) .- evaluate(-2h))) ./ (12h)
    all(isfinite, luminosity) && all(isfinite, derivative) && all(isfinite, fd) ||
        error("non-finite luminosity or derivative")
    scale = max.(abs.(luminosity), 1.0)
    all(abs.(derivative .- fd) .<= 2.0e-5 .* abs.(fd) .+ 1.0e-8 .* scale) ||
        error("Enzyme sensitivity disagrees with the finite difference")

    state = SkyMatch.generate_sfr(catalog.redshift, catalog.Mstar, params, noise)
    all(iszero, luminosity[state.qflag]) && all(iszero, derivative[state.qflag]) ||
        error("passive galaxies must have zero LIR and sensitivity")
    catalog.LIR = luminosity
    # Illustrative flux ONLY: this is not a dust SED, a real filter, or a
    # luminosity-distance conversion. It exercises the map API without tables.
    catalog.Sdemo = luminosity .* 1.0e-15 # toy scale: 1e12 Lsun -> 1 mJy
    return (; catalog, args, derivative, seed, noise_seed=seed + 1,
        max_scaled_fd_error=maximum(abs.(derivative .- fd) ./ scale))
end

read_map(path) = FITS(path, "r") do file
    read(file[1])
end

function source_checksums()
    directory = joinpath(pkgdir(SkyMatch), "src")
    files = sort(filter(name -> endswith(name, ".jl"), readdir(directory)))
    return Dict(name => bytes2hex(sha256(read(joinpath(directory, name)))) for name in files)
end

function environment_checksums()
    project = Base.active_project()
    isnothing(project) && return nothing
    manifests = (
        joinpath(dirname(project), "Manifest-v$(VERSION.major).$(VERSION.minor).toml"),
        joinpath(dirname(project), "Manifest.toml"),
    )
    manifest = findfirst(isfile, manifests)
    return Dict(
        "project_sha256" => bytes2hex(sha256(read(project))),
        "manifest_filename" => isnothing(manifest) ? nothing : basename(manifests[manifest]),
        "manifest_sha256" => isnothing(manifest) ? nothing : bytes2hex(sha256(read(manifests[manifest]))),
    )
end

"""Write toy-flux FITS maps, a PNG figure, and reproducibility metadata."""
function run_demo(case=prepare_demo(); output_dir=DEFAULT_OUTPUT)
    mkpath(output_dir)
    map_params = Dict{String,Any}(
        "filter_list" => ["demo"],
        "pixel_size" => [3.0],
        "beam_fwhm_list" => [12.0],
        "output_path" => output_dir,
        "run_name" => "synthetic",
        "sides_cat_path" => "synthetic; seed=$(case.seed); not an observed catalog",
        "gen_map_nobeam_Jy_pix" => true,
        "gen_map_smoothed_Jy_beam" => true,
    )
    SkyMatch.make_maps(case.catalog, map_params, Dict{String,Any}())
    raw = read_map(joinpath(output_dir, "synthetic_demo_nobeam_Jy_pix.fits"))
    smoothed = read_map(joinpath(output_dir, "synthetic_demo_smoothed_Jy_beam.fits"))
    isapprox(sum(raw), sum(case.catalog.Sdemo); rtol=1e-12) || error("map lost source flux")
    all(isfinite, smoothed) || error("non-finite beam map")

    fig = Figure(size=(1200, 450), fontsize=14)
    Label(fig[0, :], "SkyMatch: synthetic galaxies, illustrative flux scale", fontsize=20)
    for (column, pixels, title, unit) in (
        (1, raw, "Source map", "mJy/pixel"),
        (2, smoothed, "12 arcsec beam", "mJy/beam"),
    )
        ax = Axis(fig[1, column], title=title, xlabel="RA pixel", ylabel="Dec pixel", aspect=DataAspect())
        # FITS images use (x,y), as does Makie's heatmap matrix convention.
        hm = heatmap!(ax, log10.(1 .+ 1000 .* max.(pixels, 0.0)); colormap=:magma)
        Colorbar(fig[2, column], hm; vertical=false, label="log10(1 + flux / $unit)")
    end
    ax = Axis(fig[1, 3], title="Response to SFR scatter", xlabel="log10(LIR / solar luminosity)",
        ylabel="(dLIR / d sigma_MS) / LIR")
    active = case.catalog.LIR .> 0
    scatter!(ax, log10.(case.catalog.LIR[active]), case.derivative[active] ./ case.catalog.LIR[active];
        markersize=3, color=(:steelblue, 0.5))
    hlines!(ax, [0.0]; color=:gray, linestyle=:dash)
    Label(fig[2, 3], "Enzyme derivative; fixed galaxies and random draws", fontsize=12)
    save(joinpath(output_dir, "skymatch_synthetic_demo.png"), fig)

    metadata = Dict(
        "catalog" => "synthetic uniform sky/redshift/log-stellar-mass draws; not survey data",
        "source_count" => length(case.catalog.ra),
        "catalog_seed" => case.seed,
        "noise_seed" => case.noise_seed,
        "julia_version" => string(VERSION),
        "package_versions" => Dict(
            "SkyMatch" => string(pkgversion(SkyMatch)),
            "Enzyme" => string(pkgversion(Enzyme)),
            "CairoMakie" => string(pkgversion(CairoMakie)),
        ),
        "demo_source_sha256" => bytes2hex(sha256(read(@__FILE__))),
        "skymatch_source_sha256" => source_checksums(),
        "environment" => environment_checksums(),
        "parameters_sha256" => bytes2hex(sha256(read(PARAMETER_FILE))),
        "sigma_MS" => only(case.args[1]),
        "flux_definition" => "illustrative Sdemo[Jy] = LIR[Lsun] * 1e-15; not a modeled band",
        "pixel_arcsec" => 3.0,
        "beam_fwhm_arcsec" => 12.0,
        "derivative" => "dLIR/dsigma_MS from forward-mode Enzyme with fixed noise",
        "finite_difference_max_error_divided_by_lir" => case.max_scaled_fd_error,
        "finite_difference_passed" => true,
        "source_flux_jy" => sum(case.catalog.Sdemo),
        "map_flux_jy" => sum(raw),
    )
    open(joinpath(output_dir, "skymatch_synthetic_demo.json"), "w") do io
        JSON.print(io, metadata, 2)
        println(io)
    end
    return (; figure=fig, case, metadata, output_dir)
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) <= 1 || error("Usage: julia --project=examples examples/skymatch_demo.jl [OUTPUT_DIR]")
    result = SkyMatchDemo.run_demo(; output_dir=isempty(ARGS) ? SkyMatchDemo.DEFAULT_OUTPUT : only(ARGS))
    println("Wrote synthetic demo to ", result.output_dir)
    println("Maximum finite-difference error / LIR: ", result.case.max_scaled_fd_error)
end
