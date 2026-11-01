#!/usr/bin/env julia

ENV["CUDA_VISIBLE_DEVICES"] = "0"

using BayesMMfwd: BayesMMfwd
using Enzyme: Enzyme
using Reactant: Reactant

include("common.jl")

const LIR_SEED = 1234
const LIR_FD_STEP = 1.0e-4

"""One intrinsic infrared luminosity per galaxy, in solar luminosities.

This is the existing generate_sfr kernel followed by the same SFR2LIR
conversion as generate_dust_outputs. Only sigma_MS is active. In the current
SIDES parameterization this scatter applies to both main-sequence and
starburst galaxies; their classifications and all random draws remain fixed.
"""
function lir_per_galaxy(theta, redshift, mstar, sfr_params, noise, sfr2lir)
    varied_params = merge(sfr_params, (sigma_MS=sum(theta),))
    sfr = BayesMMfwd.generate_sfr(redshift, mstar, varied_params, noise).SFR
    return sfr2lir .* sfr
end

"""Vector dLIR_i/dsigma_MS from one forward-mode Enzyme pass (unit seed)."""
function lir_sigma_sensitivity(theta, redshift, mstar, sfr_params, noise, sfr2lir)
    return Enzyme.autodiff(
        Enzyme.Forward,
        lir_per_galaxy,
        Enzyme.Duplicated,
        Enzyme.Duplicated(theta, one.(theta)),
        Enzyme.Const(redshift),
        Enzyme.Const(mstar),
        Enzyme.Const(sfr_params),
        Enzyme.Const(noise),
        Enzyme.Const(sfr2lir),
    )[1]
end

function lir_finite_difference(args; step=LIR_FD_STEP)
    theta, tail... = args
    sigma = only(theta)
    sigma > 2step || error("finite-difference stencil requires sigma_MS > 2h")
    # Subtract before scaling to limit cancellation for bright galaxies.
    return (
        8.0 .* (lir_per_galaxy([sigma + step], tail...) .-
                lir_per_galaxy([sigma - step], tail...)) .-
        (lir_per_galaxy([sigma + 2step], tail...) .-
         lir_per_galaxy([sigma - 2step], tail...))
    ) ./ (12step)
end

function prepare_lir_case(dataset, nrows)
    raw = BayesMMfwd.load_par_file("data/SIDES_from_original.par")
    catalog = BayesMMfwd.load_sides_csv(dataset, nrows)
    redshift = Float64.(catalog.redshift)
    mstar = Float64.(catalog.Mstar)
    length(redshift) == nrows || error("catalog does not contain $nrows rows")
    params = BayesMMfwd.parse_sfr_numeric_params(raw)
    all_noise = BayesMMfwd.make_simulation_noise(nrows; seed=LIR_SEED)
    noise = (
        q_uniform=all_noise.q_uniform,
        sb_uniform=all_noise.sb_uniform,
        sfr_uniform=all_noise.sfr_uniform,
    )
    args = ([params.sigma_MS], redshift, mstar, params, noise, Float64(raw["SFR2LIR"]))
    state = BayesMMfwd.generate_sfr(redshift, mstar, params, noise)
    return (; args, qflag=state.qflag, issb=state.issb)
end

function lir_comparison(reference, candidate, lir; rtol, scaled_atol)
    length(reference) == length(candidate) == length(lir) || error("length mismatch")
    all(isfinite, reference) && all(isfinite, candidate) || error("non-finite derivative")
    max_absolute = 0.0
    max_scaled = 0.0
    max_relative = 0.0
    failures = 0
    worst_row = 1
    for i in eachindex(reference)
        absolute = abs(candidate[i] - reference[i])
        scale = max(abs(lir[i]), 1.0)
        scaled = absolute / scale
        relative = absolute / max(abs(reference[i]), scaled_atol * scale)
        if absolute > max_absolute
            max_absolute = absolute
            worst_row = i
        end
        max_scaled = max(max_scaled, scaled)
        max_relative = max(max_relative, relative)
        failures += absolute > scaled_atol * scale + rtol * abs(reference[i])
    end
    return Dict(
        "passed" => failures == 0,
        "failed_galaxies" => failures,
        "max_absolute_difference_lsun_per_dex" => max_absolute,
        "max_difference_divided_by_nominal_lir" => max_scaled,
        "max_relative_difference" => max_relative,
        "worst_absolute_row" => worst_row,
        "rtol" => rtol,
        "atol_divided_by_max_lir_1lsun" => scaled_atol,
        "acceptance" => "abs(candidate-reference) <= rtol*abs(reference) + scaled_atol*max(abs(LIR),1 Lsun)",
    )
end

function parse_lir_options(args)
    values = Dict{Symbol,Any}(
        :dataset => "data/SIDES_Bethermin2017_short.csv",
        :nrows => 1_000_000,
        :backend => "CPU",
        :results_dir => "benchmark/results",
    )
    if "--help" in args
        println("""
Usage:
  julia --startup-file=no --history-file=no --project=benchmark \\
    benchmark/benchmark_lir_sensitivity.jl --rows N --backend CPU|CUDA \\
    --results-dir benchmark/results

Computes every galaxy's dLIR/dsigma_MS at the configured sigma_MS (0.3 dex).
The first N catalog rows and seed-1234 noise are fixed. Forward-mode Enzyme
returns N derivatives, not a derivative of their sum. CPU also stores a
five-point finite-difference reference at h=1e-4 and h/2. Warm timings reuse
the existing Chairmarks/XProf harness. No paper source is edited.
""")
        return nothing
    end
    iseven(length(args)) || error("each option requires a value")
    for i in 1:2:length(args)
        key, value = args[i], args[i+1]
        if key == "--rows"
            values[:nrows] = parse(Int, value)
        elseif key == "--backend"
            values[:backend] = uppercase(value)
        elseif key == "--dataset"
            values[:dataset] = value
        elseif key == "--results-dir"
            values[:results_dir] = value
        else
            error("unknown option: $key")
        end
    end
    values[:nrows] > 0 || error("--rows must be positive")
    values[:backend] in ("CPU", "CUDA") || error("--backend must be CPU or CUDA")
    return (; values...)
end

function run_lir_sensitivity(options, command_args)
    ENV["BENCHMARK_GROUP"] = options.backend
    backend = get_backend()
    Threads.nthreads() == 1 || error("this evaluation requires one Julia thread")
    prefix = "bayesmmfwd_lir_sensitivity_$(options.nrows)"
    stem = joinpath(options.results_dir, "$(prefix)_$(backend)benchmarks")
    name = "BayesMMfwd [$(options.nrows) galaxies]/lir_sensitivity_sigma_ms"
    metadata = make_metadata_entry(merge(options, (filters=false,)), backend, command_args)
    metadata["exact_command"] = replace(
        metadata["exact_command"], "benchmark_reactant.jl" => "benchmark_lir_sensitivity.jl",
    )
    metadata["enzyme_version"] = string(Base.pkgversion(Enzyme))
    metadata["ad_mode"] = "forward; one unit tangent for sigma_MS, one derivative per galaxy"
    metadata["output_units"] = "L_sun per dex of sigma_MS"
    metadata["scope"] = "generate_sfr plus the SFR2LIR multiplication used by generate_dust_outputs"
    metadata["fixed_quantities"] = "catalog, other parameters, galaxy types, all explicit seed-1234 random draws"
    metadata["scatter_scope"] = "sigma_MS is shared by main-sequence and starburst populations"
    metadata["source_selection"] = "first N rows; no random subsampling"
    # Derivatives have luminosity units spanning many orders of magnitude;
    # do not inherit the forward-model's fixed absolute tolerance label.
    metadata["correctness_atol"] = nothing
    metadata["correctness_scaled_atol"] = 1.0e-8
    metadata["correctness_acceptance"] =
        "abs(candidate-reference) <= 2e-6*abs(reference) + 1e-8*max(abs(LIR),1 Lsun)"
    write_json(stem * "_metadata.json", metadata)

    try
        case = prepare_lir_case(options.dataset, options.nrows)
        args = case.args
        metadata["nominal_sigma_ms_dex"] = only(args[1])
        metadata["sfr2lir"] = args[end]
        nominal = lir_per_galaxy(args...)
        reference = lir_sigma_sensitivity(args...)
        all(isfinite, nominal) && all(isfinite, reference) || error("non-finite Julia result")
        all(iszero, nominal[case.qflag]) || error("passive luminosity is not zero")
        all(iszero, reference[case.qflag]) || error("passive sensitivity is not zero")

        correctness = Dict{String,Any}(
            "source_count" => options.nrows,
            "nominal_sigma_ms_dex" => only(args[1]),
            "parameter" => "sigma_MS",
            "metric" => "dLIR_i/dsigma_MS",
            "units" => "L_sun/dex",
            "passive_derivatives_exactly_zero" => true,
        )
        fd = nothing
        if backend == "CPU"
            fd_h = lir_finite_difference(args)
            fd = lir_finite_difference(args; step=LIR_FD_STEP / 2)
            for (key, baseline, candidate) in (
                ("julia_vs_finite_difference_h", fd_h, reference),
                ("julia_vs_finite_difference_half_h", fd, reference),
                ("finite_difference_step_stability", fd, fd_h),
            )
                check = lir_comparison(baseline, candidate, nominal; rtol=2.0e-5, scaled_atol=1.0e-7)
                correctness[key] = check
                check["passed"] || error("$key failed for $(check["failed_galaxies"]) galaxies")
            end
            correctness["finite_difference_stencil"] = "five-point central"
            correctness["finite_difference_steps_dex"] = [LIR_FD_STEP, LIR_FD_STEP / 2]
        end

        ra_args = Reactant.to_rarray(args)
        Reactant.synchronize(ra_args)
        results = Dict{String,Dict{String,Float64}}()
        output = run_benchmark!(
            results, backend, name, lir_sigma_sensitivity, args, ra_args;
            configs=[BenchmarkConfiguration("Default")],
        )
        candidate = Float64.(Array(output))
        check = lir_comparison(reference, candidate, nominal; rtol=2.0e-6, scaled_atol=1.0e-8)
        correctness["reactant_vs_julia"] = check
        check["passed"] || error("Reactant derivative differs at $(check["failed_galaxies"]) galaxies")
        all(iszero, candidate[case.qflag]) || error("Reactant passive sensitivity is not zero")

        values = Dict{String,Any}(
            "schema_version" => 1,
            "source_count" => options.nrows,
            "row_order" => "same as first N input catalog rows, one array element per galaxy",
            "nominal_sigma_ms_dex" => only(args[1]),
            "seed" => LIR_SEED,
            "backend" => backend,
            "reactant_d_lir_d_sigma_lsun_per_dex" => candidate,
        )
        if backend == "CPU"
            merge!(values, Dict(
                "nominal_lir_lsun" => nominal,
                "julia_d_lir_d_sigma_lsun_per_dex" => reference,
                "finite_difference_d_lir_d_sigma_lsun_per_dex" => fd,
                "qflag" => case.qflag,
                "issb" => case.issb,
            ))
        end
        # Compact column-oriented JSON preserves every galaxy without millions
        # of repeated field names; only generated summaries/plots downsample.
        mkpath(options.results_dir)
        open(stem * "_per_galaxy.json", "w") do io
            JSON.json(io, values)
            println(io)
        end
        correctness["passed"] = true
        correctness["status"] = "complete"
        save_results(results, options.results_dir, prefix, backend)
        write_json(stem * "_correctness.json", correctness)
        metadata["status"] = "complete"
        metadata["completed_at_utc"] = string(Dates.now(UTC))
        write_json(stem * "_metadata.json", metadata)
        pretty_print_results(results, "BayesMMfwd per-galaxy LIR sensitivity", backend)
        @info "LIR sensitivity validated" source_count=options.nrows backend
        return nothing
    catch exception
        metadata["status"] = "failed"
        metadata["error"] = sprint(showerror, exception)
        write_json(stem * "_metadata.json", metadata)
        write_json(stem * "_correctness.json", Dict(
            "passed" => false, "status" => "failed", "error" => metadata["error"],
        ))
        rethrow()
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    options = parse_lir_options(ARGS)
    isnothing(options) || run_lir_sensitivity(options, ARGS)
end
