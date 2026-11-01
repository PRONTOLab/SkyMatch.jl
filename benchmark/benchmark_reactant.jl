using InteractiveUtils: versioninfo

# The paper CUDA resource is physical device 0 only. Set visibility before
# loading Reactant so XLA cannot initialize the second installed GPU.
ENV["CUDA_VISIBLE_DEVICES"] = "0"

include("forward_model.jl")

function print_benchmark_usage(io=stdout)
    println(
        io,
        """
Usage:
  julia --project=benchmark benchmark/benchmark_reactant.jl [options]

Options:
  --dataset PATH       SIDES CSV input (default: data/SIDES_Bethermin2017_short2.csv)
  --rows N             Read only the first N rows
  --large              Use the large CSV (all rows unless followed by --rows)
  --filters            Include filter-grid fluxes (requires every FILTER_NAME.h5)
  --backend NAME       Reactant backend: CPU, CUDA, or TPU
                       (default: BENCHMARK_GROUP, then Reactant auto-detection)
  --results-dir PATH   JSON output directory (default: benchmark/results)
  --help               Show this message

Results use Reactant.jl's Runtime (s) and TFLOP/s schema. CPU runtime uses
Chairmarks for both Julia and Reactant; accelerator runtime and FLOP/s use
XProf. A separate memory JSON records host allocations and XProf backend-memory
peaks. Ordinary Julia is recorded only for CPU runs; every backend is checked
against it. The 1,000,000-row invocation also records placement, compilation,
complete materialization, and one-shot overheads. The CPU invocation also
records native Julia first-call and JIT-compilation time. Every successful
invocation records field-by-field correctness extrema.
"""
    )
end

function parse_benchmark_command_line(args)
    options = (
        dataset="data/SIDES_Bethermin2017_short2.csv",
        nrows=nothing,
        filters=false,
        backend=nothing,
        results_dir=joinpath(@__DIR__, "results"),
    )
    values = Dict{Symbol,Any}(pairs(options))

    index = 1
    while index <= length(args)
        argument = args[index]
        if argument == "--help"
            print_benchmark_usage()
            return nothing
        elseif argument == "--large"
            values[:dataset] = "data/SIDES_Bethermin2017_short.csv"
            values[:nrows] = nothing
        elseif argument == "--filters"
            values[:filters] = true
        elseif argument in ("--dataset", "--rows", "--backend", "--results-dir")
            index == length(args) &&
                throw(ArgumentError("$argument requires a value"))
            value = args[index+1]
            if argument == "--dataset"
                values[:dataset] = value
            elseif argument == "--rows"
                values[:nrows] = parse(Int, value)
                values[:nrows] > 0 ||
                    throw(ArgumentError("--rows must be positive"))
            elseif argument == "--backend"
                group = uppercase(value)
                group == "GPU" && (group = "CUDA")
                group in ("CPU", "CUDA", "TPU") || throw(ArgumentError(
                    "--backend must be CPU, CUDA, or TPU",
                ))
                values[:backend] = group
            else
                values[:results_dir] = value
            end
            index += 1
        else
            throw(ArgumentError("unknown argument: $argument"))
        end
        index += 1
    end
    return (; values...)
end

function benchmark_failure_status(exception)
    message = lowercase(sprint(showerror, exception))
    oom_phrases = (
        "out of memory",
        "resource_exhausted",
        "resource exhausted",
        "cuda_error_out_of_memory",
        "failed to allocate",
    )
    any(phrase -> occursin(phrase, message), oom_phrases) && return "oom"
    correctness_phrases = (
        "differs between julia and reactant",
        "mismatched dimensions",
        "non-finite reactant values",
        "different output fields",
    )
    any(phrase -> occursin(phrase, message), correctness_phrases) &&
        return "correctness_failed"
    return "failed"
end

function run_all_benchmarks(options, args=ARGS)
    if !isnothing(options.backend)
        ENV["BENCHMARK_GROUP"] = options.backend
    end
    backend = get_backend()
    backend == "CPU" && Threads.nthreads() != 1 && error(
        "paper Julia CPU benchmarks require exactly one Julia thread; " *
        "found $(Threads.nthreads())",
    )
    @info sprint(io -> versioninfo(io; verbose=true))

    metadata = make_metadata_entry(options, backend, args)
    save_benchmark_metadata(options.results_dir, metadata)

    results = Dict{String,Dict{String,Float64}}()
    try
        measurement = run_bayesmmfwd_benchmark!(
            results,
            backend;
            dataset=options.dataset,
            nrows=options.nrows,
            filters=options.filters,
        )

        n_gal = measurement.n_gal
        metadata["source_count"] = n_gal
        mode = options.filters ? "forward_filters" : "forward"
        prefix = "bayesmmfwd_$(mode)_$n_gal"
        save_results(results, options.results_dir, prefix, backend)
        if !isnothing(measurement.overheads)
            save_overhead_results(
                measurement.overheads,
                measurement.julia_overheads,
                options.results_dir,
                prefix,
                backend,
                measurement.benchmark_name,
            )
        end
        save_correctness_results(
            measurement.correctness,
            options.results_dir,
            prefix,
            backend,
            measurement.benchmark_name,
        )

        metadata["status"] = "complete"
        metadata["completed_at_utc"] = string(Dates.now(UTC))
        save_benchmark_metadata(options.results_dir, metadata)
        pretty_print_results(results, "BayesMMfwd", backend)
        return results
    catch exception
        status = benchmark_failure_status(exception)
        source_count = metadata["source_count"]
        metadata["status"] = status
        metadata["error"] = sprint(showerror, exception)
        metadata["completed_at_utc"] = string(Dates.now(UTC))
        save_benchmark_metadata(options.results_dir, metadata)

        if source_count > 0
            mode = options.filters ? "forward_filters" : "forward"
            prefix = "bayesmmfwd_$(mode)_$source_count"
            benchmark_name = "BayesMMfwd [$source_count galaxies]/$mode"
            save_failed_correctness(
                options.results_dir,
                prefix,
                backend,
                benchmark_name,
                status,
                metadata["error"],
            )
        end

        if status == "oom"
            @error "Benchmark recorded an out-of-memory result" backend source_count exception
            return Dict{String,Dict{String,Float64}}()
        end
        rethrow()
    end
end

function benchmark_main(args=ARGS)
    options = parse_benchmark_command_line(args)
    isnothing(options) && return nothing
    return run_all_benchmarks(options, args)
end

if abspath(PROGRAM_FILE) == @__FILE__
    benchmark_main()
end
