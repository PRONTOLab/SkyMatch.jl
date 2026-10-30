# Reduce XLA / TensorFlow logging
ENV["TF_CPP_MIN_LOG_LEVEL"] = "3"
ENV["ABSL_MIN_LOG_LEVEL"] = "3"
ENV["GLOG_minloglevel"] = "2"
ENV["JULIA_DEBUG"] = ""

using InteractiveUtils: versioninfo
using Random
using Reactant

# Set GPU visibility before loading Reactant.
ENV["CUDA_VISIBLE_DEVICES"] = "0"

# Reuse the exact fixture and model implementation from your NUTS script.
# Change this path if your working NUTS script has a different filename.
const NUTS_SCRIPT = joinpath(@__DIR__, "clean_forward_model_in_impulse.jl")

isfile(NUTS_SCRIPT) || error(
    "Could not find NUTS script at $NUTS_SCRIPT. " *
    "Update NUTS_SCRIPT to point to your working script."
)

include(NUTS_SCRIPT)
include(joinpath(@__DIR__, "..", "benchmark", "common.jl"))

function run_synthetic_benchmark!(
    results::Dict,
    backend::String,
    benchmark_name::String,
    fn::F,
    cpu_args::Tuple,
    ra_args::Tuple;
    skip_cpu::Bool=false,
    configs::Vector{BenchmarkConfiguration},
    benchmark_seconds::Float64=5.0,
    benchmark_samples::Int=100,
) where {F}
    for metric in (
        "Runtime (s)",
        "TFLOP/s",
        "Host allocated bytes",
        "Host allocations",
        "Backend peak memory (bytes)",
    )
        if !haskey(results, metric)
            results[metric] = Dict{String,Float64}()
        end
    end

    # Run CPU/Julia benchmark if on CPU backend
    if backend == "CPU" && !skip_cpu
        full_benchmark_name = string(benchmark_name, "/CPU/Julia")

        # Warmup
        fn(cpu_args...)

        # Benchmark using Chairmarks
        bench = @b fn($(cpu_args)...) seconds = benchmark_seconds evals = 1 samples =
            benchmark_samples

        results["Runtime (s)"][full_benchmark_name] = bench.time
        results["TFLOP/s"][full_benchmark_name] = -1.0
        results["Host allocated bytes"][full_benchmark_name] = Float64(bench.bytes)
        results["Host allocations"][full_benchmark_name] = Float64(bench.allocs)

        print_stmt = @sprintf(
            "%100s     :     %.5gs     %.5g MiB     %.0f allocations",
            full_benchmark_name,
            bench.time,
            bench.bytes / 2.0^20,
            bench.allocs,
        )
        @info print_stmt
        GC.gc(true)
    end

    # Run Reactant for each configuration. Chairmarks gives a directly
    # comparable CPU measurement; accelerators use synchronized XProf timing.
    output = nothing
    for config in configs
        full_benchmark_name = string(benchmark_name, "/", backend, "/", config.name)

        compile_options = something(config.compile_options, Reactant.CompileOptions())
        compile_options =
            Reactant.__compile_options_with_updated_sync(compile_options, true)

        if backend == "CPU"
            compiled_fn = Reactant.compile(fn, ra_args; compile_options)
            output = compiled_fn(ra_args...)
            allocation_bench = @b $compiled_fn($(ra_args)...) seconds =
                benchmark_seconds evals = 1 samples = benchmark_samples
            results["Runtime (s)"][full_benchmark_name] = allocation_bench.time
            results["TFLOP/s"][full_benchmark_name] = -1.0
        else
            prof_results = Reactant.Profiler.profile_with_xprof(
                fn,
                ra_args...;
                nrepeat=config.nrepeat,
                compile_options=compile_options,
            )
            output = prof_results.val
            results["Runtime (s)"][full_benchmark_name] =
                prof_results.profiling_result.runtime_ns / 1e9
            metrics = prof_results.profiling_result.metrics_data
            tflops = isnothing(metrics) ? -1.0 : metrics.raw_flops_rate / 1e12
            results["TFLOP/s"][full_benchmark_name] =
                isfinite(tflops) ? tflops : -1.0

            memory_data = prof_results.profiling_result.memory_data
            for allocator in sort!(collect(keys(memory_data)))
                memory_name = string(full_benchmark_name, " [", allocator, "]")
                peak_bytes = memory_data[allocator].peak_bytes_usage_lifetime
                results["Backend peak memory (bytes)"][memory_name] =
                    Float64(peak_bytes)
                @info "Reactant backend memory peak" benchmark =
                    full_benchmark_name allocator peak_bytes
            end

            output = nothing
            prof_results = nothing
            GC.gc(true)
            compiled_fn = Reactant.compile(fn, ra_args; compile_options)
            output = compiled_fn(ra_args...)
            allocation_bench = @b $compiled_fn($(ra_args)...) seconds =
                benchmark_seconds evals = 1 samples = 1
        end

        results["Host allocated bytes"][full_benchmark_name] =
            Float64(allocation_bench.bytes)
        results["Host allocations"][full_benchmark_name] =
            Float64(allocation_bench.allocs)

        print_stmt = @sprintf(
            "%100s     :     %.5gs     %.5g TFLOP/s     %.5g MiB     %.0f allocations",
            full_benchmark_name,
            results["Runtime (s)"][full_benchmark_name],
            results["TFLOP/s"][full_benchmark_name],
            allocation_bench.bytes / 2.0^20,
            allocation_bench.allocs,
        )
        @info print_stmt
        GC.gc(true)
    end

    return output
end


# ============================================================
# COMMAND LINE
# ============================================================

function print_usage()
    println("""
Usage:
  julia --project=benchmark benchmark/benchmark_reactant_synthetic.jl [options]

Options:
  --rows N          Number of synthetic galaxies (default: 1000)
  --backend NAME    CPU, CUDA, or TPU (default: CUDA)
  --seed N          Synthetic input seed (default: BENCHMARK_SEED + 1)
  --results-dir P   Results directory (default: benchmark/results_synthetic)
  --help            Show this message

Examples:
  julia --project=benchmark benchmark/benchmark_reactant_synthetic.jl --rows 1000 --backend CUDA
  julia --project=benchmark benchmark/benchmark_reactant_synthetic.jl --rows 10000 --backend CUDA
""")
end

function parse_options(args)
    options = Dict{Symbol,Any}(
        :rows => 1000,
        :backend => "CUDA",
        :seed => BENCHMARK_SEED + 1,
        :results_dir => joinpath(@__DIR__, "results_synthetic"),
    )

    i = 1
    while i <= length(args)
        arg = args[i]

        if arg == "--help"
            print_usage()
            return nothing
        end

        i < length(args) || error("$arg requires a value")
        value = args[i + 1]

        if arg == "--rows"
            options[:rows] = parse(Int, value)
            options[:rows] > 0 || error("--rows must be positive")
        elseif arg == "--backend"
            backend = uppercase(value)
            backend == "GPU" && (backend = "CUDA")
            backend in ("CPU", "CUDA", "TPU") ||
                error("--backend must be CPU, CUDA, or TPU")
            options[:backend] = backend
        elseif arg == "--seed"
            options[:seed] = parse(Int, value)
        elseif arg == "--results-dir"
            options[:results_dir] = value
        else
            error("Unknown option: $arg")
        end

        i += 2
    end

    return (; options...)
end


# ============================================================
# SYNTHETIC INPUT GENERATION
# ============================================================

function make_synthetic_inputs(n; seed=BENCHMARK_SEED + 1)
    # Match generate_forward_model_data() in the NUTS script.
    rng = MersenneTwister(seed)

    redshift = 3.0 .* rand(rng, n)
    mstar = 10.0 .^ (9.0 .+ 3.0 .* rand(rng, n))

    inputs = (
        redshift,
        Mstar=mstar,
        dlum_mpc=100.0 .* (1.0 .+ redshift),
        dlum_m=100.0 .* (1.0 .+ redshift) .* MPC_TO_M,
    )

    return inputs
end

function make_synthetic_benchmark_args(n; seed=BENCHMARK_SEED + 1)
    fixture = synthetic_fixture()

    inputs = make_synthetic_inputs(n; seed)
    noise = make_simulation_noise(n)

    # Use the same base model parameters as the NUTS experiment.
    params = fixture.params

    return (inputs, params, noise)
end


# ============================================================
# OUTPUT MATERIALIZATION AND CORRECTNESS
# ============================================================

function materialize_output(value::NamedTuple)
    names = keys(value)
    values = map(materialize_output, Tuple(value))
    return NamedTuple{names}(values)
end

materialize_output(value::Tuple) = map(materialize_output, value)
materialize_output(value::AbstractArray) = Array(value)
materialize_output(value) = value

function force_host_completion(value::NamedTuple)
    foreach(force_host_completion, Tuple(value))
    return nothing
end

function force_host_completion(value::Tuple)
    foreach(force_host_completion, value)
    return nothing
end

function force_host_completion(value::AbstractArray)
    isempty(value) || (value[firstindex(value)]; value[lastindex(value)])
    return nothing
end

force_host_completion(value) = nothing

function check_forward_output(reference, candidate; rtol=2e-6, atol=1e-8)
    keys(reference) == keys(candidate) ||
        error("Julia and Reactant returned different output fields")

    max_absolute_difference = 0.0
    max_relative_difference = 0.0
    worst_field = "none"

    for name in keys(reference)
        expected = getproperty(reference, name)
        actual = getproperty(candidate, name)

        if expected === nothing
            actual === nothing ||
                error("$name differs between Julia and Reactant")
            continue
        end

        size(expected) == size(actual) ||
            error("$name has mismatched dimensions")

        if eltype(expected) <: Bool
            expected == actual ||
                error("$name differs between Julia and Reactant")
            continue
        end

        all(isfinite, actual) ||
            error("$name contains non-finite Reactant values")

        isapprox(expected, actual; rtol, atol) ||
            error("$name differs between Julia and Reactant")

        for i in eachindex(expected, actual)
            a = Float64(expected[i])
            b = Float64(actual[i])

            abs_diff = abs(a - b)
            rel_diff = abs_diff / max(abs(a), atol)

            max_absolute_difference =
                max(max_absolute_difference, abs_diff)

            if rel_diff > max_relative_difference
                max_relative_difference = rel_diff
                worst_field = string(name)
            end
        end
    end

    @info "Correctness check passed" max_absolute_difference max_relative_difference worst_field rtol atol

    return (;
        passed=true,
        max_absolute_difference,
        max_relative_difference,
        worst_field,
        rtol,
        atol,
    )
end


# ============================================================
# BENCHMARK
# ============================================================

function run_synthetic_benchmark(options)
    ENV["BENCHMARK_GROUP"] = options.backend
    backend = get_backend()

    backend == "CPU" && Threads.nthreads() != 1 &&
        error("CPU benchmark requires exactly one Julia thread")

    @info sprint(io -> versioninfo(io; verbose=true))

    n = options.rows

    println("\n========== SYNTHETIC FORWARD MODEL BENCHMARK ==========")
    println("Number of galaxies: ", n)
    println("Backend: ", backend)
    println("Seed: ", options.seed)

    cpu_args = make_synthetic_benchmark_args(
        n;
        seed=options.seed,
    )

    inputs, params, noise = cpu_args

    @assert length(inputs.redshift) == n
    @assert length(inputs.Mstar) == n
    @assert length(noise.q_uniform) == n

    println("Generated ", n, " synthetic galaxies.")
    println("Redshift range: ", extrema(inputs.redshift))
    println("Stellar mass range: ", extrema(inputs.Mstar))

    # Convert the same inputs to Reactant arrays.
    ra_args = Reactant.to_rarray(cpu_args)
    Reactant.synchronize(ra_args)

    # Benchmark the Reactant forward model.
    results = Dict{String,Dict{String,Float64}}()
    benchmark_name = "SkyMatch Synthetic [$n galaxies]/forward"

    reactant_output = run_synthetic_benchmark!(
        results,
        backend,
        benchmark_name,
        forward_model,
        cpu_args,
        ra_args;
        configs=[BenchmarkConfiguration("Default")],
    )

    # Run the ordinary Julia implementation on identical inputs.
    println("\nChecking Julia vs. Reactant correctness...")
    reference = forward_model(cpu_args...)

    candidate = materialize_output(reactant_output)
    force_host_completion(candidate)

    correctness = check_forward_output(reference, candidate)

    # Save benchmark results.
    mkpath(options.results_dir)

    prefix = "skymatch_synthetic_forward_$n"

    save_results(
        results,
        options.results_dir,
        prefix,
        backend,
    )

    correctness_for_saving = merge(
        correctness,
        (worst_output_field = correctness.worst_field,)
    )

    save_correctness_results(
        correctness_for_saving,
        options.results_dir,
        prefix,
        backend,
        benchmark_name,
    )

    println("\n========== BENCHMARK COMPLETE ==========")
    pretty_print_results(results, "SkyMatch Synthetic", backend)

    println("Results saved in: ", options.results_dir)

    return (; results, correctness, n, backend)
end


# ============================================================
# MAIN
# ============================================================

function main(args=ARGS)
    options = parse_options(args)
    isnothing(options) && return nothing

    return run_synthetic_benchmark(options)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
