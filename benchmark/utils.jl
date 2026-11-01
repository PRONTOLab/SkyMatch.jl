# Adapted from Reactant.jl/benchmark/utils.jl so this benchmark can move into
# Reactant's benchmark suite without changing its result format.

using Dates: Dates, UTC
using JSON: JSON
using Libdl: Libdl
using PrettyTables: pretty_table

function standardized_results(values::Dict{String,Float64}, unit::String)
    results = Vector{Dict{String,Union{String,Float64}}}()
    for name in sort!(collect(keys(values)))
        push!(results, Dict("name" => name, "value" => values[name], "unit" => unit))
    end
    return results
end


function write_json(path::String, value)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON.json(io, value; pretty=true)
        println(io)
    end
    return path
end

function metric_rows(name::String, metrics)
    rows = Vector{Dict{String,Any}}()
    for (metric, value, unit) in metrics
        push!(
            rows,
            Dict(
                "name" => name,
                "metric" => metric,
                "value" => value,
                "unit" => unit,
            ),
        )
    end
    return rows
end

function save_overhead_results(
    overheads,
    julia_overheads,
    results_dir::String,
    prefix::String,
    backend::String,
    benchmark_name::String,
)
    name = string(benchmark_name, "/", backend, "/Default")
    metrics = (
        ("Input placement", overheads.input_placement_seconds, "s"),
        ("Compilation", overheads.compile_seconds, "s"),
        ("First synchronized execution", overheads.first_execution_seconds, "s"),
        ("Resident execution", overheads.resident_execution_seconds, "s"),
        ("Output materialization", overheads.materialization_seconds, "s"),
        ("One-shot total", overheads.one_shot_seconds, "s"),
    )
    path = joinpath(
        results_dir,
        string(prefix, "_", backend, "benchmarks_overheads.json"),
    )
    rows = metric_rows(name, metrics)
    if !isnothing(julia_overheads)
        julia_name = string(benchmark_name, "/CPU/Julia")
        julia_metrics = (
            ("Input placement", julia_overheads.input_placement_seconds, "s"),
            ("Compilation", julia_overheads.compile_seconds, "s"),
            (
                "First synchronized execution",
                julia_overheads.first_execution_seconds,
                "s",
            ),
            (
                "Resident execution",
                julia_overheads.resident_execution_seconds,
                "s",
            ),
            (
                "Output materialization",
                julia_overheads.materialization_seconds,
                "s",
            ),
            ("One-shot total", julia_overheads.one_shot_seconds, "s"),
        )
        append!(rows, metric_rows(julia_name, julia_metrics))
    end
    write_json(path, rows)
    @info "Saved overhead results" overheads = path
    return path
end

function save_correctness_results(
    correctness,
    results_dir::String,
    prefix::String,
    backend::String,
    benchmark_name::String,
)
    name = string(benchmark_name, "/", backend, "/Default")
    metrics = (
        ("Pass", correctness.passed, "boolean"),
        (
            "Maximum absolute difference",
            correctness.max_absolute_difference,
            "absolute",
        ),
        (
            "Maximum relative difference",
            correctness.max_relative_difference,
            "relative",
        ),
        ("Worst output field", correctness.worst_output_field, "field"),
        ("rtol", correctness.rtol, "relative"),
        ("atol", correctness.atol, "absolute"),
        ("Status", "complete", "status"),
    )
    path = joinpath(
        results_dir,
        string(prefix, "_", backend, "benchmarks_correctness.json"),
    )
    write_json(path, metric_rows(name, metrics))
    @info "Saved correctness results" correctness = path
    return path
end

function save_failed_correctness(
    results_dir::String,
    prefix::String,
    backend::String,
    benchmark_name::String,
    status::String,
    message::String,
)
    name = string(benchmark_name, "/", backend, "/Default")
    metrics = Vector{Tuple{String,Any,String}}([
        ("Status", status, "status"),
        ("Error", message, "message"),
        ("rtol", 2.0e-6, "relative"),
        ("atol", 1.0e-8, "absolute"),
    ])
    status == "correctness_failed" &&
        pushfirst!(metrics, ("Pass", false, "boolean"))
    path = joinpath(
        results_dir,
        string(prefix, "_", backend, "benchmarks_correctness.json"),
    )
    write_json(path, metric_rows(name, metrics))
    @info "Saved failed correctness status" correctness = path status
    return path
end

function command_output(command::Cmd)
    try
        return strip(read(command, String))
    catch
        return nothing
    end
end

function parse_os_name()
    path = "/etc/os-release"
    isfile(path) || return string(Sys.KERNEL)
    for line in eachline(path)
        startswith(line, "PRETTY_NAME=") || continue
        return strip(split(line, "="; limit=2)[2], '"')
    end
    return string(Sys.KERNEL)
end

function parse_nvidia_metadata()
    query = Cmd([
        "nvidia-smi",
        "--id=0",
        "--query-gpu=index,name,memory.total,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ])
    output = command_output(query)
    isnothing(output) && return Dict{String,Any}()

    fields = strip.(split(first(split(output, '\n')), ','))
    length(fields) == 5 || return Dict{String,Any}()
    total_mib = tryparse(Int, fields[3])
    free_mib = tryparse(Int, fields[4])
    metadata = Dict{String,Any}(
        "selected_gpu_index" => tryparse(Int, fields[1]),
        "gpu_name" => fields[2],
        "nvidia_driver_version" => fields[5],
    )
    if !isnothing(total_mib)
        metadata["gpu_usable_memory_mib"] = total_mib
        metadata["gpu_usable_memory_bytes"] = total_mib * 1024^2
    end
    if !isnothing(free_mib)
        metadata["gpu_free_memory_mib_at_start"] = free_mib
        metadata["gpu_free_memory_bytes_at_start"] = free_mib * 1024^2
    end
    return metadata
end

function format_cudnn_version(version::Integer)
    major = version ÷ 10_000
    minor = (version % 10_000) ÷ 100
    patch = version % 100
    return "$major.$minor.$patch"
end

function format_cudart_version(version::Integer)
    major = version ÷ 1000
    minor = (version % 1000) ÷ 10
    return "$major.$minor"
end

function reactant_gpu_runtime_versions()
    runtime_module = nothing
    for (package, module_value) in Base.loaded_modules
        if package.name == "Reactant_jll"
            runtime_module = module_value
            break
        end
    end
    isnothing(runtime_module) && return Dict{String,Any}()
    isdefined(runtime_module, :libcudnn) || return Dict{String,Any}()

    library = getproperty(runtime_module, :libcudnn)
    try
        handle = Libdl.dlopen(library)
        cudnn_symbol = Libdl.dlsym(handle, :cudnnGetVersion)
        cudart_symbol = Libdl.dlsym(handle, :cudnnGetCudartVersion)
        cudnn = ccall(cudnn_symbol, Csize_t, ())
        cudart = ccall(cudart_symbol, Csize_t, ())
        return Dict{String,Any}(
            "cudnn_version" => format_cudnn_version(cudnn),
            "cuda_runtime_version" => format_cudart_version(cudart),
        )
    catch
        return Dict{String,Any}()
    end
end

function collect_hardware_metadata()
    cpu_info = Sys.cpu_info()
    hardware = Dict{String,Any}(
        "os" => parse_os_name(),
        "kernel" => something(
            command_output(Cmd(["uname", "-srmo"])),
            string(Sys.KERNEL),
        ),
        "cpu_model" => isempty(cpu_info) ? "unknown" : cpu_info[1].model,
        "logical_cpu_threads" => Sys.CPU_THREADS,
        "ram_bytes" => Sys.total_memory(),
        "julia_threads" => Threads.nthreads(),
    )
    merge!(hardware, parse_nvidia_metadata())
    merge!(hardware, reactant_gpu_runtime_versions())
    return hardware
end

function exact_benchmark_command(args)
    suffix = isempty(args) ? "" : string(" ", join(args, " "))
    return string(
        "julia --startup-file=no --history-file=no --project=benchmark ",
        "benchmark/benchmark_reactant.jl",
        suffix,
    )
end

function make_metadata_entry(options, backend::String, args)
    source_count = isnothing(options.nrows) ? 0 : options.nrows
    estimator =
        backend == "CPU" ? "Chairmarks scalar bench.time" :
        "synchronized XProf mean runtime"
    sample_count = backend == "CPU" ? 100 : 25
    selected_device =
        backend == "CPU" ? "Reactant/XLA default CPU device" :
        "physical NVIDIA GPU 0 (CUDA_VISIBLE_DEVICES=0)"

    return Dict{String,Any}(
        "source_commit" => something(
            command_output(Cmd(["git", "rev-parse", "HEAD"])),
            "unknown",
        ),
        "dirty" => !isempty(
            something(
                command_output(Cmd(["git", "status", "--porcelain=v1"])),
                "",
            ),
        ),
        "exact_command" => exact_benchmark_command(args),
        "julia_version" => string(VERSION),
        "reactant_version" => string(Base.pkgversion(Reactant)),
        "internal_estimator" => estimator,
        "internal_executions_or_samples" => sample_count,
        "chairmarks_budget_seconds" => 5.0,
        "chairmarks_evaluations_per_sample" => 1,
        "chairmarks_requested_samples" => 100,
        "xprof_synchronized_executions" => 25,
        "seed" => 1234,
        "filters_enabled" => options.filters,
        "dataset" => options.dataset,
        "source_count" => source_count,
        "cpu_thread_policy" => Dict(
            "julia" => "one Julia thread",
            "reactant_cpu" => "XLA default CPU parallelization",
        ),
        "selected_backend" => backend,
        "selected_device" => selected_device,
        "selected_gpu_index" => backend == "CUDA" ? 0 : nothing,
        "hardware" => collect_hardware_metadata(),
        "correctness_rtol" => 2.0e-6,
        "correctness_atol" => 1.0e-8,
        "status" => "running",
        "recorded_at_utc" => string(Dates.now(UTC)),
    )
end

function save_benchmark_metadata(results_dir::String, entry::Dict{String,Any})
    path = joinpath(results_dir, "benchmark_metadata.json")
    document = if isfile(path)
        JSON.parsefile(path)
    else
        Dict{String,Any}(
            "schema_version" => 1,
            "correctness" => Dict("rtol" => 2.0e-6, "atol" => 1.0e-8),
            "invocations" => Any[],
        )
    end

    invocations = document["invocations"]
    filter!(invocations) do previous
        previous["selected_backend"] != entry["selected_backend"] ||
            previous["source_count"] != entry["source_count"]
    end
    push!(invocations, entry)
    sort!(
        invocations;
        by=invocation -> (
            invocation["source_count"],
            invocation["selected_backend"],
        ),
    )
    write_json(path, document)
    return path
end

function get_backend()
    benchmark_group = get(ENV, "BENCHMARK_GROUP", nothing)

    if benchmark_group == "CUDA"
        Reactant.set_default_backend("gpu")
        @info "Running CUDA benchmarks" maxlog = 1
    elseif benchmark_group == "TPU"
        Reactant.set_default_backend("tpu")
        @info "Running TPU benchmarks" maxlog = 1
    elseif benchmark_group == "CPU"
        Reactant.set_default_backend("cpu")
        @info "Running CPU benchmarks" maxlog = 1
    else
        benchmark_group = String(split(string(first(Reactant.devices())), ":")[1])
        @info "Running $(benchmark_group) benchmarks" maxlog = 1
    end

    @assert benchmark_group in ("CPU", "CUDA", "TPU") "Unknown backend: $(benchmark_group)"
    return benchmark_group
end

function save_results(
    results::Dict{String,Dict{String,Float64}},
    results_dir::String,
    prefix::String,
    backend::String,
)
    mkpath(results_dir)

    benchmark_filename = string(prefix, "_", backend, "benchmarks.json")
    tflops_filename = string(prefix, "_", backend, "benchmarks_tflops.json")
    memory_filename = string(prefix, "_", backend, "benchmarks_memory.json")
    benchmark_filepath = joinpath(results_dir, benchmark_filename)
    tflops_filepath = joinpath(results_dir, tflops_filename)
    memory_filepath = joinpath(results_dir, memory_filename)

    runtime_results = standardized_results(results["Runtime (s)"], "s")
    tflops_results = standardized_results(results["TFLOP/s"], "TFLOP/s")
    memory_results = Vector{Dict{String,Union{String,Float64}}}()
    memory_metrics = (
        ("Host allocated bytes", "B"),
        ("Host allocations", "allocations"),
        ("Backend peak memory (bytes)", "B"),
    )
    for (metric, unit) in memory_metrics
        values = get(results, metric, Dict{String,Float64}())
        for name in sort!(collect(keys(values)))
            push!(
                memory_results,
                Dict(
                    "name" => name,
                    "metric" => metric,
                    "value" => values[name],
                    "unit" => unit,
                ),
            )
        end
    end

    open(benchmark_filepath, "w") do io
        JSON.json(io, runtime_results; pretty=true)
    end
    open(tflops_filepath, "w") do io
        JSON.json(io, tflops_results; pretty=true)
    end
    open(memory_filepath, "w") do io
        JSON.json(io, memory_results; pretty=true)
    end

    @info "Saved benchmark results" runtime = benchmark_filepath tflops =
        tflops_filepath memory = memory_filepath
    return benchmark_filepath, tflops_filepath, memory_filepath
end

function pretty_print_results(results::Dict, suite::String, backend::String)
    runtime_results = get(results, "Runtime (s)", Dict{String,Float64}())
    tflops_results = get(results, "TFLOP/s", Dict{String,Float64}())
    byte_results = get(results, "Host allocated bytes", Dict{String,Float64}())
    allocation_results = get(results, "Host allocations", Dict{String,Float64}())

    if isempty(runtime_results)
        @warn "No benchmark results to display for $(suite)/$(backend)"
        return nothing
    end

    sorted_keys = sort(collect(keys(runtime_results)))
    table = Matrix{Any}(undef, length(sorted_keys), 8)
    for (index, key) in enumerate(sorted_keys)
        parts = split(key, "/")
        while length(parts) < 4
            push!(parts, "")
        end
        table[index, 1] = parts[1]
        table[index, 2] = parts[2]
        table[index, 3] = parts[3]
        table[index, 4] = join(parts[4:end], "/")
        table[index, 5] = runtime_results[key]
        table[index, 6] = get(tflops_results, key, -1.0)
        table[index, 7] = round(Int, get(byte_results, key, 0.0))
        table[index, 8] = round(Int, get(allocation_results, key, 0.0))
    end

    println()
    println("="^120)
    println("  Benchmark Results: $(suite) / $(backend)")
    println("="^120)
    pretty_table(
        table;
        alignment=[:l, :l, :l, :l, :c, :c, :c, :c],
        column_labels=[
            "Benchmark",
            "Mode",
            "Backend",
            "Config",
            "Time (s)",
            "TFLOP/s",
            "Host allocated (B)",
            "Host allocations",
        ],
        display_size=(-1, -1),
    )
    println("="^120)
    return nothing
end
