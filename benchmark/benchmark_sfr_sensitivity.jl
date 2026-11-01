#!/usr/bin/env julia

# Keep the paper GPU policy consistent with benchmark_reactant.jl.
ENV["CUDA_VISIBLE_DEVICES"] = "0"

using BayesMMfwd: BayesMMfwd
using Enzyme: Enzyme
using Reactant: Reactant

include("common.jl")

const SENSITIVITY_SEED = 1234
const FD_STEP = 1.0e-4
const AD_RTOL = 2.0e-6
const AD_ATOL = 1.0e-10
const FD_RTOL = 2.0e-5

"""
Mean log-scaled generated SFR as a function of the one-element `a0` array.

Catalog inputs, stochastic draws, and `sfr_scale` are constants during
differentiation. `sum(theta)` avoids scalar indexing and remains equivalent to
`theta[1]` for the required one-element active parameter array.
"""
function sfr_summary_objective(
    theta,
    redshift,
    mstar,
    sfr_params,
    noise,
    sfr_scale,
)
    varied_params = merge(sfr_params, (a0=sum(theta),))
    sfr = BayesMMfwd.generate_sfr(
        redshift,
        mstar,
        varied_params,
        noise,
    ).SFR
    return sum(log1p.(sfr ./ sfr_scale)) / length(sfr)
end

"""Reverse-mode derivative of `sfr_summary_objective` with respect to `theta`."""
function sfr_a0_sensitivity(
    theta,
    redshift,
    mstar,
    sfr_params,
    noise,
    sfr_scale,
)
    return Enzyme.gradient(
        Enzyme.Reverse,
        sfr_summary_objective,
        theta,
        Enzyme.Const(redshift),
        Enzyme.Const(mstar),
        Enzyme.Const(sfr_params),
        Enzyme.Const(noise),
        Enzyme.Const(sfr_scale),
    )[1]
end

function five_point_sensitivity(args; step=FD_STEP)
    theta, tail... = args
    a0 = only(theta)
    evaluate(delta) = sfr_summary_objective([a0 + delta], tail...)
    return (
        -evaluate(2step) + 8evaluate(step) - 8evaluate(-step) +
        evaluate(-2step)
    ) / (12step)
end

function prepare_sensitivity_case(dataset::String, nrows::Int)
    raw_params = BayesMMfwd.load_par_file("data/SIDES_from_original.par")
    catalog = BayesMMfwd.load_sides_csv(dataset, nrows)
    redshift = Float64.(catalog.redshift)
    mstar = Float64.(catalog.Mstar)
    n_gal = length(redshift)
    n_gal == nrows || error("requested $nrows rows but loaded $n_gal")

    sfr_params = BayesMMfwd.parse_sfr_numeric_params(raw_params)
    all_noise = BayesMMfwd.make_simulation_noise(n_gal; seed=SENSITIVITY_SEED)
    # The isolated SFR kernel consumes exactly these three random arrays.
    noise = (
        q_uniform=all_noise.q_uniform,
        sb_uniform=all_noise.sb_uniform,
        sfr_uniform=all_noise.sfr_uniform,
    )
    theta = [sfr_params.a0]
    nominal_sfr = BayesMMfwd.generate_sfr(
        redshift,
        mstar,
        sfr_params,
        noise,
    ).SFR
    sfr_scale = sum(nominal_sfr) / n_gal
    isfinite(sfr_scale) && sfr_scale > 0 || error(
        "nominal mean SFR must be finite and positive; got $sfr_scale",
    )
    return (theta, redshift, mstar, sfr_params, noise, sfr_scale)
end

function scalar_gradient(value)
    array = value isa AbstractArray ? Array(value) : value
    length(array) == 1 || error("expected one a0 derivative, got $(length(array))")
    result = Float64(only(array))
    isfinite(result) || error("a0 sensitivity is not finite: $result")
    return result
end

function write_sensitivity_correctness(
    results_dir,
    prefix,
    backend,
    name;
    n_gal,
    objective,
    nominal_a0,
    sfr_scale,
    julia_sensitivity,
    reactant_sensitivity,
    finite_difference=nothing,
)
    absolute_difference = abs(reactant_sensitivity - julia_sensitivity)
    relative_difference = absolute_difference / max(abs(julia_sensitivity), AD_ATOL)
    rows = metric_rows(
        name,
        [
            ("Status", "complete", "status"),
            ("Pass", true, "boolean"),
            ("Source count", n_gal, "galaxies"),
            ("Seed", SENSITIVITY_SEED, "integer"),
            ("Parameter", "a0", "name"),
            ("Nominal a0", nominal_a0, "dimensionless"),
            ("Objective", "mean(log1p(SFR / sfr_scale))", "definition"),
            ("Nominal objective", objective, "dimensionless"),
            ("SFR scale", sfr_scale, "same units as SFR"),
            ("Julia Enzyme sensitivity", julia_sensitivity, "dobjective/da0"),
            (
                "Reactant Enzyme sensitivity",
                reactant_sensitivity,
                "dobjective/da0",
            ),
            ("Maximum absolute difference", absolute_difference, "absolute"),
            ("Maximum relative difference", relative_difference, "relative"),
            ("AD comparison rtol", AD_RTOL, "relative"),
            ("AD comparison atol", AD_ATOL, "absolute"),
        ],
    )
    if !isnothing(finite_difference)
        fd_absolute = abs(julia_sensitivity - finite_difference)
        fd_relative = fd_absolute / max(abs(finite_difference), AD_ATOL)
        append!(
            rows,
            metric_rows(
                name,
                [
                    ("Finite-difference sensitivity", finite_difference, "dobjective/da0"),
                    ("Finite-difference stencil", "five-point central", "method"),
                    ("Finite-difference step", FD_STEP, "a0"),
                    ("Julia-vs-FD absolute difference", fd_absolute, "absolute"),
                    ("Julia-vs-FD relative difference", fd_relative, "relative"),
                    ("Finite-difference comparison rtol", FD_RTOL, "relative"),
                ],
            ),
        )
    end
    path = joinpath(
        results_dir,
        string(prefix, "_", backend, "benchmarks_correctness.json"),
    )
    write_json(path, rows)
    return path
end

function parse_options(args)
    values = Dict{Symbol,Any}(
        :dataset => "data/SIDES_Bethermin2017_short.csv",
        :nrows => 1024,
        :backend => nothing,
        :results_dir => "benchmark/results",
    )
    index = 1
    while index <= length(args)
        argument = args[index]
        if argument == "--help"
            println("""
Usage:
  julia --project=benchmark benchmark/benchmark_sfr_sensitivity.jl \\
    --rows N --backend CPU|CUDA --results-dir benchmark/results

Measures the warm reverse-mode derivative of mean(log1p(SFR / sfr_scale))
with respect to a0. Inputs and explicit seed-1234 noise remain fixed.
""")
            return nothing
        elseif argument in ("--dataset", "--rows", "--backend", "--results-dir")
            index == length(args) && error("$argument requires a value")
            value = args[index+1]
            if argument == "--dataset"
                values[:dataset] = value
            elseif argument == "--rows"
                values[:nrows] = parse(Int, value)
                values[:nrows] > 0 || error("--rows must be positive")
            elseif argument == "--backend"
                backend = uppercase(value)
                backend == "GPU" && (backend = "CUDA")
                backend in ("CPU", "CUDA") || error(
                    "--backend must be CPU or CUDA",
                )
                values[:backend] = backend
            else
                values[:results_dir] = value
            end
            index += 1
        else
            error("unknown argument: $argument")
        end
        index += 1
    end
    isnothing(values[:backend]) && error("--backend CPU|CUDA is required")
    return (; values...)
end

function run_sensitivity_benchmark(options)
    ENV["BENCHMARK_GROUP"] = options.backend
    backend = get_backend()
    backend == "CPU" && Threads.nthreads() != 1 && error(
        "Julia sensitivity timing requires one Julia thread; found $(Threads.nthreads())",
    )

    cpu_args = prepare_sensitivity_case(options.dataset, options.nrows)
    n_gal = length(cpu_args[2])
    benchmark_name = "BayesMMfwd [$n_gal galaxies]/sfr_sensitivity_a0"
    prefix = "bayesmmfwd_sfr_sensitivity_$n_gal"
    nominal_objective = sfr_summary_objective(cpu_args...)

    # Enzyme compilation/JIT is excluded from every reported warm derivative.
    julia_sensitivity = scalar_gradient(sfr_a0_sensitivity(cpu_args...))
    ra_args = Reactant.to_rarray(cpu_args)
    Reactant.synchronize(ra_args)
    compile_options = Reactant.__compile_options_with_updated_sync(
        Reactant.CompileOptions(),
        true,
    )
    compile_start = time_ns()
    compiled = Reactant.compile(
        sfr_a0_sensitivity,
        ra_args;
        compile_options,
    )
    compile_seconds = (time_ns() - compile_start) / 1.0e9
    compiled_output = compiled(ra_args...)
    Reactant.synchronize(compiled_output)

    results = Dict{String,Dict{String,Float64}}()
    reactant_output = run_benchmark!(
        results,
        backend,
        benchmark_name,
        sfr_a0_sensitivity,
        cpu_args,
        ra_args;
        configs=[BenchmarkConfiguration("Default")],
    )
    reactant_sensitivity = scalar_gradient(reactant_output)
    isapprox(
        reactant_sensitivity,
        julia_sensitivity;
        rtol=AD_RTOL,
        atol=AD_ATOL,
    ) || error(
        "Reactant sensitivity $reactant_sensitivity differs from Julia $julia_sensitivity",
    )

    finite_difference = n_gal == 1_000_000 ? five_point_sensitivity(cpu_args) : nothing
    if !isnothing(finite_difference)
        isapprox(
            julia_sensitivity,
            finite_difference;
            rtol=FD_RTOL,
            atol=AD_ATOL,
        ) || error(
            "Julia Enzyme sensitivity $julia_sensitivity differs from five-point " *
            "finite difference $finite_difference",
        )
    end

    save_results(results, options.results_dir, prefix, backend)
    overhead_path = joinpath(
        options.results_dir,
        string(prefix, "_", backend, "benchmarks_overheads.json"),
    )
    write_json(
        overhead_path,
        metric_rows(benchmark_name, [("Compilation", compile_seconds, "s")]),
    )
    correctness_path = write_sensitivity_correctness(
        options.results_dir,
        prefix,
        backend,
        benchmark_name;
        n_gal,
        objective=nominal_objective,
        nominal_a0=only(cpu_args[1]),
        sfr_scale=cpu_args[end],
        julia_sensitivity,
        reactant_sensitivity,
        finite_difference,
    )
    @info "Sensitivity benchmark complete" backend n_gal compile_seconds correctness_path
    pretty_print_results(results, "BayesMMfwd SFR sensitivity", backend)
    return results
end

function main(args=ARGS)
    options = parse_options(args)
    isnothing(options) && return nothing
    return run_sensitivity_benchmark(options)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
