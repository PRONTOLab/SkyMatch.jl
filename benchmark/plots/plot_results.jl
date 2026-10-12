#!/usr/bin/env julia

using CairoMakie
using JSON
using Printf

const SOURCE_COUNTS = [1_024, 100_000, 1_000_000, 5_584_998]
const SOURCE_TICK_LABELS = ["1,024", "100k", "1M", "5.58M"]
const JULIA_LABEL = "Julia CPU (1 thread)"
const CPU_LABEL = "Reactant CPU (XLA default)"
const GPU_LABEL = "Reactant GPU (1x RTX 5090)"
const IMPLEMENTATION_LABELS = [JULIA_LABEL, CPU_LABEL, GPU_LABEL]
const COLORS = Dict(
    JULIA_LABEL => RGBf(0.0, 0.0, 0.0),
    CPU_LABEL => RGBf(0.0, 114 / 255, 178 / 255),
    GPU_LABEL => RGBf(213 / 255, 85 / 255, 0.0),
)
const MARKERS = Dict(
    JULIA_LABEL => :circle,
    CPU_LABEL => :rect,
    GPU_LABEL => :utriangle,
)
const LINESTYLES = Dict(
    JULIA_LABEL => :solid,
    CPU_LABEL => :dash,
    GPU_LABEL => :dot,
)
const SUMMARY_COLUMNS = [
    "git_commit",
    "dirty",
    "backend",
    "hardware_label",
    "source_count",
    "internal_estimator",
    "internal_executions_or_samples",
    "filters_enabled",
    "seed",
    "runtime_seconds",
    "input_placement_seconds",
    "compile_seconds",
    "materialization_seconds",
    "one_shot_seconds",
    "host_allocated_bytes",
    "backend_peak_bytes",
    "correctness_passed",
    "max_absolute_difference",
    "max_relative_difference",
    "worst_output_field",
    "status",
]
const AXIS_STYLE = (
    xlabelsize=8.5,
    ylabelsize=8.5,
    xticklabelsize=7.5,
    yticklabelsize=7.5,
    xgridcolor=(:black, 0.12),
    ygridcolor=(:black, 0.12),
    xgridwidth=0.6,
    ygridwidth=0.6,
    xminorgridvisible=false,
    yminorgridvisible=false,
    topspinevisible=false,
    rightspinevisible=false,
)

function parse_options(args)
    options = Dict(
        "--results-dir" => "benchmark/results",
        "--output-dir" => "benchmark/plots/generated",
    )
    if "--help" in args
        println(
            "Usage: julia --project=benchmark benchmark/plots/plot_results.jl " *
            "--results-dir DIR --output-dir DIR",
        )
        exit()
    end

    iseven(length(args)) || error("every plotting option requires a value")
    index = 1
    while index <= length(args)
        option = args[index]
        haskey(options, option) || error(
            "unknown option: $option; use --results-dir and --output-dir",
        )
        options[option] = args[index+1]
        index += 2
    end
    return options
end

const LEGACY_METRIC_PREFIXES = ("bayesmmfwd_forward_", "bayesmm_forward_")

function metric_path(results_dir, source_count, backend, suffix)
    canonical = joinpath(
        results_dir,
        "skymatch_forward_$(source_count)_$(backend)$(suffix)",
    )
    legacy = [
        joinpath(
            results_dir,
            "$(prefix)$(source_count)_$(backend)$(suffix)",
        ) for prefix in LEGACY_METRIC_PREFIXES
    ]
    filter!(isfile, legacy)
    if isfile(canonical)
        isempty(legacy) || @warn(
            "Both current and legacy benchmark files exist; using SkyMatch",
            canonical,
            legacy,
        )
        return canonical
    end
    return isempty(legacy) ? nothing : first(legacy)
end

function read_metric(results_dir, source_count, backend, suffix)
    path = metric_path(results_dir, source_count, backend, suffix)
    return isnothing(path) ? nothing : JSON.parsefile(path)
end

function benchmark_value(records, backend, configuration)
    isnothing(records) && return nothing
    ending = "/$backend/$configuration"
    for record in records
        endswith(record["name"], ending) || continue
        value = record["value"]
        return value isa Number && isfinite(value) && value > 0 ?
               Float64(value) : nothing
    end
    return nothing
end

function memory_value(records, backend, configuration, metric)
    isnothing(records) && return nothing
    name_fragment = "/$backend/$configuration"
    values = Float64[]
    for record in records
        get(record, "metric", nothing) == metric || continue
        occursin(name_fragment, record["name"]) || continue
        value = record["value"]
        value isa Number && isfinite(value) && value > 0 || continue
        push!(values, Float64(value))
    end
    return isempty(values) ? nothing : maximum(values)
end

function named_metrics(records, backend)
    values = Dict{String,Any}()
    isnothing(records) && return values
    name_fragment = "/$backend/Default"
    for record in records
        occursin(name_fragment, record["name"]) || continue
        values[record["metric"]] = record["value"]
    end
    return values
end

function read_metadata(results_dir)
    path = joinpath(results_dir, "benchmark_metadata.json")
    isfile(path) || error(
        "missing $path; rerun the benchmark invocations so provenance and " *
        "failure status are recorded",
    )
    document = JSON.parsefile(path)
    haskey(document, "invocations") || error("$path has no invocations array")
    index = Dict{Tuple{Int,String},Dict{String,Any}}()
    for invocation in document["invocations"]
        source_count = Int(invocation["source_count"])
        backend = String(invocation["selected_backend"])
        key = (source_count, backend)
        haskey(index, key) && error(
            "duplicate metadata entries for backend=$backend, source_count=$source_count",
        )
        index[key] = Dict{String,Any}(invocation)
    end
    return document, index
end

metadata_value(metadata, key) =
    isnothing(metadata) ? nothing : get(metadata, key, nothing)

function correctness_values(metrics, status)
    status == "oom" && return (nothing, nothing, nothing, nothing)
    passed = get(metrics, "Pass", nothing)
    max_absolute = get(metrics, "Maximum absolute difference", nothing)
    max_relative = get(metrics, "Maximum relative difference", nothing)
    worst_field = get(metrics, "Worst output field", nothing)
    return passed, max_absolute, max_relative, worst_field
end

function result_status(metadata, runtime, correctness_passed; reactant)
    declared = metadata_value(metadata, "status")
    declared in ("oom", "failed", "correctness_failed", "unsupported") &&
        return String(declared)
    isnothing(runtime) && return "missing"
    if reactant
        correctness_passed === true && return "complete"
        correctness_passed === false && return "correctness_failed"
        return "missing"
    end
    return declared == "complete" ? "complete" : "missing"
end

function base_row(metadata, backend, hardware_label, source_count)
    return Dict{String,Any}(
        "git_commit" => metadata_value(metadata, "source_commit"),
        "dirty" => metadata_value(metadata, "dirty"),
        "backend" => backend,
        "hardware_label" => hardware_label,
        "source_count" => source_count,
        "internal_estimator" => metadata_value(metadata, "internal_estimator"),
        "internal_executions_or_samples" =>
            metadata_value(metadata, "internal_executions_or_samples"),
        "filters_enabled" => metadata_value(metadata, "filters_enabled"),
        "seed" => metadata_value(metadata, "seed"),
        "runtime_seconds" => nothing,
        "input_placement_seconds" => nothing,
        "compile_seconds" => nothing,
        "materialization_seconds" => nothing,
        "one_shot_seconds" => nothing,
        "host_allocated_bytes" => nothing,
        "backend_peak_bytes" => nothing,
        "correctness_passed" => nothing,
        "max_absolute_difference" => nothing,
        "max_relative_difference" => nothing,
        "worst_output_field" => nothing,
        "status" => "missing",
    )
end

function build_rows(results_dir, metadata_index)
    rows = Vector{Dict{String,Any}}()
    for source_count in SOURCE_COUNTS
        cpu_metadata = get(metadata_index, (source_count, "CPU"), nothing)
        cpu_runtime = read_metric(
            results_dir,
            source_count,
            "CPU",
            "benchmarks.json",
        )
        cpu_memory = read_metric(
            results_dir,
            source_count,
            "CPU",
            "benchmarks_memory.json",
        )
        cpu_overheads = named_metrics(
            read_metric(
                results_dir,
                source_count,
                "CPU",
                "benchmarks_overheads.json",
            ),
            "CPU",
        )
        cpu_correctness = named_metrics(
            read_metric(
                results_dir,
                source_count,
                "CPU",
                "benchmarks_correctness.json",
            ),
            "CPU",
        )

        julia_runtime = benchmark_value(cpu_runtime, "CPU", "Julia")
        julia_row = base_row(
            cpu_metadata,
            "Julia",
            JULIA_LABEL,
            source_count,
        )
        julia_row["runtime_seconds"] = julia_runtime
        julia_row["host_allocated_bytes"] = memory_value(
            cpu_memory,
            "CPU",
            "Julia",
            "Host allocated bytes",
        )
        julia_row["status"] = result_status(
            cpu_metadata,
            julia_runtime,
            nothing;
            reactant=false,
        )
        push!(rows, julia_row)

        cpu_passed, cpu_max_absolute, cpu_max_relative, cpu_worst =
            correctness_values(
                cpu_correctness,
                metadata_value(cpu_metadata, "status"),
            )
        reactant_cpu_runtime = benchmark_value(cpu_runtime, "CPU", "Default")
        cpu_row = base_row(
            cpu_metadata,
            "Reactant CPU",
            CPU_LABEL,
            source_count,
        )
        cpu_row["runtime_seconds"] = reactant_cpu_runtime
        cpu_row["input_placement_seconds"] =
            get(cpu_overheads, "Input placement", nothing)
        cpu_row["compile_seconds"] = get(cpu_overheads, "Compilation", nothing)
        cpu_row["materialization_seconds"] =
            get(cpu_overheads, "Output materialization", nothing)
        cpu_row["one_shot_seconds"] =
            get(cpu_overheads, "One-shot total", nothing)
        cpu_row["host_allocated_bytes"] = memory_value(
            cpu_memory,
            "CPU",
            "Default",
            "Host allocated bytes",
        )
        cpu_row["correctness_passed"] = cpu_passed
        cpu_row["max_absolute_difference"] = cpu_max_absolute
        cpu_row["max_relative_difference"] = cpu_max_relative
        cpu_row["worst_output_field"] = cpu_worst
        cpu_row["status"] = result_status(
            cpu_metadata,
            reactant_cpu_runtime,
            cpu_passed;
            reactant=true,
        )
        push!(rows, cpu_row)

        cuda_metadata = get(metadata_index, (source_count, "CUDA"), nothing)
        cuda_runtime = read_metric(
            results_dir,
            source_count,
            "CUDA",
            "benchmarks.json",
        )
        cuda_memory = read_metric(
            results_dir,
            source_count,
            "CUDA",
            "benchmarks_memory.json",
        )
        cuda_overheads = named_metrics(
            read_metric(
                results_dir,
                source_count,
                "CUDA",
                "benchmarks_overheads.json",
            ),
            "CUDA",
        )
        cuda_correctness = named_metrics(
            read_metric(
                results_dir,
                source_count,
                "CUDA",
                "benchmarks_correctness.json",
            ),
            "CUDA",
        )
        cuda_passed, cuda_max_absolute, cuda_max_relative, cuda_worst =
            correctness_values(
                cuda_correctness,
                metadata_value(cuda_metadata, "status"),
            )
        reactant_gpu_runtime =
            benchmark_value(cuda_runtime, "CUDA", "Default")
        gpu_row = base_row(
            cuda_metadata,
            "Reactant GPU",
            GPU_LABEL,
            source_count,
        )
        gpu_row["runtime_seconds"] = reactant_gpu_runtime
        gpu_row["input_placement_seconds"] =
            get(cuda_overheads, "Input placement", nothing)
        gpu_row["compile_seconds"] =
            get(cuda_overheads, "Compilation", nothing)
        gpu_row["materialization_seconds"] =
            get(cuda_overheads, "Output materialization", nothing)
        gpu_row["one_shot_seconds"] =
            get(cuda_overheads, "One-shot total", nothing)
        gpu_row["host_allocated_bytes"] = memory_value(
            cuda_memory,
            "CUDA",
            "Default",
            "Host allocated bytes",
        )
        gpu_row["backend_peak_bytes"] = memory_value(
            cuda_memory,
            "CUDA",
            "Default",
            "Backend peak memory (bytes)",
        )
        gpu_row["correctness_passed"] = cuda_passed
        gpu_row["max_absolute_difference"] = cuda_max_absolute
        gpu_row["max_relative_difference"] = cuda_max_relative
        gpu_row["worst_output_field"] = cuda_worst
        gpu_row["status"] = result_status(
            cuda_metadata,
            reactant_gpu_runtime,
            cuda_passed;
            reactant=true,
        )
        push!(rows, gpu_row)
    end

    length(rows) == 3 * length(SOURCE_COUNTS) ||
        error("internal error while constructing canonical summary rows")
    return rows
end

function row_for(rows, hardware_label, source_count)
    for row in rows
        row["hardware_label"] == hardware_label || continue
        row["source_count"] == source_count || continue
        return row
    end
    error("missing canonical row for $hardware_label at $source_count sources")
end

function valid_runtime(row)
    return row["status"] == "complete" &&
           row["runtime_seconds"] isa Number &&
           row["runtime_seconds"] > 0
end

function gpu_capacity_bytes(metadata_document)
    capacities = Int[]
    for invocation in metadata_document["invocations"]
        invocation["selected_backend"] == "CUDA" || continue
        hardware = get(invocation, "hardware", Dict{String,Any}())
        capacity = get(hardware, "gpu_usable_memory_bytes", nothing)
        capacity isa Number || continue
        push!(capacities, Int(capacity))
    end
    isempty(capacities) && error(
        "CUDA metadata does not contain hardware-queried usable GPU memory",
    )
    length(unique(capacities)) > 1 && @warn(
        "CUDA invocations recorded different usable-memory capacities",
        capacities,
    )
    return first(capacities)
end

function log_limits(values; padding=0.22)
    positive = Float64[value for value in values if value isa Number && value > 0]
    isempty(positive) && error("cannot construct a logarithmic axis without data")
    low = minimum(positive)
    high = maximum(positive)
    low == high && return (low / 3, high * 3)
    span = log10(high) - log10(low)
    return (
        10.0^(log10(low) - padding * span),
        10.0^(log10(high) + padding * span),
    )
end

function add_panel_label!(axis, label)
    text!(
        axis,
        0.025,
        0.975;
        text=label,
        space=:relative,
        align=(:left, :top),
        fontsize=9,
        font=:bold,
    )
end

function plot_series!(axis, rows, hardware_label, value_key; label=nothing)
    values = Float64[]
    for source_count in SOURCE_COUNTS
        row = row_for(rows, hardware_label, source_count)
        value = row[value_key]
        push!(
            values,
            row["status"] == "complete" && value isa Number ?
            Float64(value) : NaN,
        )
    end
    options = (
        color=COLORS[hardware_label],
        linestyle=LINESTYLES[hardware_label],
        marker=MARKERS[hardware_label],
        linewidth=1.8,
        markersize=6.5,
    )
    if isnothing(label)
        scatterlines!(axis, Float64.(SOURCE_COUNTS), values; options...)
    else
        scatterlines!(
            axis,
            Float64.(SOURCE_COUNTS),
            values;
            options...,
            label,
        )
    end
    return values
end

function add_oom_marker!(axis, source_count, y)
    scatter!(
        axis,
        [Float64(source_count)],
        [Float64(y)];
        color=COLORS[GPU_LABEL],
        marker=:xcross,
        markersize=8,
        strokewidth=1.8,
    )
    text!(
        axis,
        Float64(source_count),
        Float64(y);
        text="OOM",
        color=COLORS[GPU_LABEL],
        align=(:right, :bottom),
        offset=(-2, 4),
        fontsize=7.5,
        font=:bold,
    )
end

function save_vector_pair(figure, output_dir, stem)
    mkpath(output_dir)
    pdf_path = joinpath(output_dir, "$stem.pdf")
    svg_path = joinpath(output_dir, "$stem.svg")
    save(pdf_path, figure; pt_per_unit=1)
    save(svg_path, figure; px_per_unit=1)
    println("wrote $pdf_path")
    println("wrote $svg_path")
end

function make_figure2(output_dir, rows)
    figure = Figure(size=(515, 205), fontsize=8)
    runtime_axis = Axis(
        figure[2, 1];
        AXIS_STYLE...,
        xscale=log10,
        yscale=log10,
        xlabel="Number of galaxies, N_src",
        ylabel="Warm execution time (s)",
        xticks=(Float64.(SOURCE_COUNTS), SOURCE_TICK_LABELS),
    )
    speedup_axis = Axis(
        figure[2, 2];
        AXIS_STYLE...,
        xscale=log10,
        yscale=log10,
        xlabel="Number of galaxies, N_src",
        ylabel="Speedup over Julia CPU (1 thread)",
        xticks=(Float64.(SOURCE_COUNTS), SOURCE_TICK_LABELS),
    )

    runtime_values = Float64[]
    for hardware_label in IMPLEMENTATION_LABELS
        append!(
            runtime_values,
            filter(
                isfinite,
                plot_series!(
                    runtime_axis,
                    rows,
                    hardware_label,
                    "runtime_seconds";
                    label=hardware_label,
                ),
            ),
        )
    end

    speedup_values = Float64[1.0]
    for hardware_label in (CPU_LABEL, GPU_LABEL)
        values = Float64[]
        for source_count in SOURCE_COUNTS
            julia_row = row_for(rows, JULIA_LABEL, source_count)
            backend_row = row_for(rows, hardware_label, source_count)
            value = valid_runtime(julia_row) && valid_runtime(backend_row) ?
                    julia_row["runtime_seconds"] /
                    backend_row["runtime_seconds"] : NaN
            push!(values, value)
            isfinite(value) && push!(speedup_values, value)
        end
        scatterlines!(
            speedup_axis,
            Float64.(SOURCE_COUNTS),
            values;
            color=COLORS[hardware_label],
            linestyle=LINESTYLES[hardware_label],
            marker=MARKERS[hardware_label],
            linewidth=1.8,
            markersize=6.5,
        )
    end
    hlines!(
        speedup_axis,
        [1.0];
        color=(:black, 0.55),
        linestyle=:dash,
        linewidth=1.0,
    )
    text!(
        speedup_axis,
        1_100.0,
        1.0;
        text="1x",
        align=(:left, :bottom),
        offset=(0, 2),
        fontsize=7.5,
    )

    runtime_limits = log_limits(runtime_values)
    speedup_limits = log_limits(speedup_values)
    ylims!(runtime_axis, runtime_limits...)
    ylims!(speedup_axis, speedup_limits...)
    xlims!(runtime_axis, 760, 8.0e6)
    xlims!(speedup_axis, 760, 8.0e6)

    gpu_largest = row_for(rows, GPU_LABEL, last(SOURCE_COUNTS))
    if gpu_largest["status"] == "oom"
        runtime_oom_y = 10.0^(
            log10(runtime_limits[1]) +
            0.92 * (log10(runtime_limits[2]) - log10(runtime_limits[1]))
        )
        speedup_oom_y = 10.0^(
            log10(speedup_limits[1]) +
            0.92 * (log10(speedup_limits[2]) - log10(speedup_limits[1]))
        )
        add_oom_marker!(runtime_axis, last(SOURCE_COUNTS), runtime_oom_y)
        add_oom_marker!(speedup_axis, last(SOURCE_COUNTS), speedup_oom_y)
    end

    add_panel_label!(runtime_axis, "(a)")
    add_panel_label!(speedup_axis, "(b)")
    Legend(
        figure[1, 1:2],
        runtime_axis;
        orientation=:horizontal,
        nbanks=1,
        framevisible=false,
        labelsize=7.5,
        patchsize=(13, 8),
        colgap=10,
        tellheight=true,
    )
    colgap!(figure.layout, 12)
    rowgap!(figure.layout, 2)
    save_vector_pair(figure, output_dir, "fig2_runtime_speedup")
end

function require_figure3_rows(rows)
    selected = Dict{String,Dict{String,Any}}()
    for hardware_label in (JULIA_LABEL, CPU_LABEL, GPU_LABEL)
        row = row_for(rows, hardware_label, 1_000_000)
        valid_runtime(row) || error(
            "Figure 3 requires a valid 1,000,000-source $hardware_label " *
            "runtime; rerun its canonical benchmark command",
        )
        if hardware_label != JULIA_LABEL
            for key in (
                "input_placement_seconds",
                "compile_seconds",
                "materialization_seconds",
                "one_shot_seconds",
            )
                row[key] isa Number && row[key] > 0 || error(
                    "Figure 3 requires $key in " *
                    "skymatch_forward_1000000_" *
                    (hardware_label == CPU_LABEL ? "CPU" : "CUDA") *
                    "benchmarks_overheads.json",
                )
            end
        end
        selected[hardware_label] = row
    end
    return selected
end

function smallest_break_even(compile_seconds, recurring_seconds, julia_seconds)
    recurring_seconds < julia_seconds || return nothing
    threshold = compile_seconds / (julia_seconds - recurring_seconds)
    candidate = max(1, floor(Int, threshold) + 1)
    return candidate <= 100_000 ? candidate : nothing
end

function calculate_break_evens(rows)
    selected = require_figure3_rows(rows)
    julia_seconds = selected[JULIA_LABEL]["runtime_seconds"]
    values = Vector{Dict{String,Any}}()
    for hardware_label in (CPU_LABEL, GPU_LABEL)
        row = selected[hardware_label]
        resident = row["runtime_seconds"]
        materialized =
            row["input_placement_seconds"] + resident +
            row["materialization_seconds"]
        for (workflow, recurring) in (
            ("resident", resident),
            ("materialized catalog", materialized),
        )
            push!(
                values,
                Dict(
                    "backend" => hardware_label,
                    "workflow" => workflow,
                    "break_even_k" => smallest_break_even(
                        row["compile_seconds"],
                        recurring,
                        julia_seconds,
                    ),
                ),
            )
        end
    end
    return values
end

function figure3_component_label(component_key, seconds)
    if component_key == "input_placement_seconds"
        return @sprintf("%.3f s", seconds)
    elseif component_key == "compile_seconds"
        return @sprintf("%.2f s", seconds)
    elseif component_key == "runtime_seconds"
        return seconds < 0.01 ?
               @sprintf("%.3f ms", 1_000 * seconds) :
               @sprintf("%.4f s", seconds)
    elseif component_key == "materialization_seconds"
        return @sprintf("%.4f s", seconds)
    end
    error("unknown Figure 3 component: $component_key")
end

function checked_figure3_break_evens(selected, break_evens)
    calculated = Dict{Tuple{String,String},Int}()
    for result in break_evens
        break_even = result["break_even_k"]
        break_even isa Integer || error(
            "Figure 3 requires a finite integer break-even for " *
            "$(result["backend"]) $(result["workflow"])",
        )
        calculated[(result["backend"], result["workflow"])] = Int(break_even)
    end

    expected = Dict(
        (CPU_LABEL, "resident") => 31,
        (GPU_LABEL, "resident") => 28,
        (CPU_LABEL, "materialized catalog") => 61,
        (GPU_LABEL, "materialized catalog") => 69,
    )
    julia_seconds = selected[JULIA_LABEL]["runtime_seconds"]
    for (key, expected_break_even) in expected
        actual = get(calculated, key, nothing)
        @assert(
            actual == expected_break_even,
            "Figure 3 break-even changed for $(first(key)) $(last(key)): " *
            "expected $expected_break_even, got $actual",
        )

        hardware_label, workflow = key
        row = selected[hardware_label]
        recurring_seconds = workflow == "resident" ?
                            row["runtime_seconds"] :
                            row["input_placement_seconds"] +
                            row["runtime_seconds"] +
                            row["materialization_seconds"]
        compile_seconds = row["compile_seconds"]
        @assert compile_seconds + actual * recurring_seconds <
                actual * julia_seconds
        if actual > 1
            previous = actual - 1
            @assert !(
                compile_seconds + previous * recurring_seconds <
                previous * julia_seconds
            )
        end
    end
    return calculated
end

function make_figure3(output_dir, rows, break_evens)
    selected = require_figure3_rows(rows)
    checked_break_evens = checked_figure3_break_evens(selected, break_evens)
    figure = Figure(size=(515, 214), fontsize=8)

    component_names = [
        "Input placement",
        "Compilation",
        "Resident execution",
        "Output materialization",
    ]
    component_keys = [
        "input_placement_seconds",
        "compile_seconds",
        "runtime_seconds",
        "materialization_seconds",
    ]
    component_y = Float64[4, 3, 2, 1]
    overhead_axis = Axis(
        figure[2, 1];
        AXIS_STYLE...,
        xscale=log10,
        xlabel="Time (s)",
        yticks=(component_y, component_names),
    )

    workflow_names = ["Resident workflow", "Host-output workflow"]
    workflow_keys = ["resident", "materialized catalog"]
    workflow_y = Float64[2, 1]
    break_even_axis = Axis(
        figure[2, 2];
        AXIS_STYLE...,
        xlabel="Break-even execution count, K",
        xticks=0:10:70,
        yticks=(workflow_y, workflow_names),
    )

    backend_y_offset = Dict(CPU_LABEL => 0.14, GPU_LABEL => -0.14)
    component_values = Float64[]
    for (component_index, component_key) in enumerate(component_keys)
        base_y = component_y[component_index]
        cpu_value = Float64(selected[CPU_LABEL][component_key])
        gpu_value = Float64(selected[GPU_LABEL][component_key])
        append!(component_values, (cpu_value, gpu_value))
        lines!(
            overhead_axis,
            [cpu_value, gpu_value],
            [
                base_y + backend_y_offset[CPU_LABEL],
                base_y + backend_y_offset[GPU_LABEL],
            ];
            color=(:black, 0.18),
            linewidth=0.8,
        )
        for (hardware_label, value) in (
            (CPU_LABEL, cpu_value),
            (GPU_LABEL, gpu_value),
        )
            y = base_y + backend_y_offset[hardware_label]
            label_on_left = component_key == "compile_seconds"
            scatter!(
                overhead_axis,
                [value],
                [y];
                color=COLORS[hardware_label],
                marker=MARKERS[hardware_label],
                markersize=6.5,
                strokecolor=:white,
                strokewidth=0.5,
            )
            text!(
                overhead_axis,
                value,
                y;
                text=figure3_component_label(component_key, value),
                color=COLORS[hardware_label],
                align=(label_on_left ? :right : :left, :center),
                offset=(label_on_left ? -5 : 5, 0),
                fontsize=7.5,
            )
        end
    end
    xlims!(overhead_axis, log_limits(component_values; padding=0.22)...)
    ylims!(overhead_axis, 0.5, 4.5)

    for (workflow_index, workflow_key) in enumerate(workflow_keys)
        base_y = workflow_y[workflow_index]
        cpu_break_even = checked_break_evens[(CPU_LABEL, workflow_key)]
        gpu_break_even = checked_break_evens[(GPU_LABEL, workflow_key)]
        lines!(
            break_even_axis,
            Float64[cpu_break_even, gpu_break_even],
            [
                base_y + backend_y_offset[CPU_LABEL],
                base_y + backend_y_offset[GPU_LABEL],
            ];
            color=(:black, 0.18),
            linewidth=0.8,
        )
        for (hardware_label, break_even) in (
            (CPU_LABEL, cpu_break_even),
            (GPU_LABEL, gpu_break_even),
        )
            y = base_y + backend_y_offset[hardware_label]
            scatter!(
                break_even_axis,
                [Float64(break_even)],
                [y];
                color=COLORS[hardware_label],
                marker=MARKERS[hardware_label],
                markersize=6.5,
                strokecolor=:white,
                strokewidth=0.5,
            )
            right_side = break_even >= 66
            text!(
                break_even_axis,
                Float64(break_even),
                y;
                text="K=$break_even",
                color=COLORS[hardware_label],
                align=(right_side ? :right : :left, :center),
                offset=(right_side ? -5 : 5, 0),
                fontsize=7.5,
            )
        end
    end
    xlims!(break_even_axis, 0, 75)
    ylims!(break_even_axis, 0.5, 2.5)

    legend_elements = [
        MarkerElement(
            color=COLORS[CPU_LABEL],
            marker=MARKERS[CPU_LABEL],
            markersize=6.5,
        ),
        MarkerElement(
            color=COLORS[GPU_LABEL],
            marker=MARKERS[GPU_LABEL],
            markersize=6.5,
        ),
    ]
    Legend(
        figure[1, 1:2],
        legend_elements,
        [CPU_LABEL, GPU_LABEL];
        orientation=:horizontal,
        nbanks=1,
        framevisible=false,
        labelsize=7.5,
        patchsize=(13, 8),
        colgap=14,
        tellheight=true,
    )

    add_panel_label!(overhead_axis, "(a)")
    add_panel_label!(break_even_axis, "(b)")
    colgap!(figure.layout, 12)
    rowgap!(figure.layout, 1)
    save_vector_pair(figure, output_dir, "fig3_overheads_amortization")
end

function make_figure4(output_dir, rows, usable_gpu_bytes)
    figure = Figure(size=(515, 205), fontsize=8)
    gpu_axis = Axis(
        figure[2, 1];
        AXIS_STYLE...,
        xscale=log10,
        xlabel="Number of galaxies, N_src",
        ylabel="Peak GPU allocator memory (GiB)",
        xticks=(Float64.(SOURCE_COUNTS), SOURCE_TICK_LABELS),
    )
    host_axis = Axis(
        figure[2, 2];
        AXIS_STYLE...,
        xscale=log10,
        yscale=log10,
        xlabel="Number of galaxies, N_src",
        ylabel="Host bytes allocated per warm call",
        xticks=(Float64.(SOURCE_COUNTS), SOURCE_TICK_LABELS),
    )

    gpu_peak_gib = Float64[]
    for source_count in SOURCE_COUNTS
        row = row_for(rows, GPU_LABEL, source_count)
        value = row["backend_peak_bytes"]
        push!(
            gpu_peak_gib,
            row["status"] == "complete" && value isa Number ?
            Float64(value) / 2.0^30 : NaN,
        )
    end
    scatterlines!(
        gpu_axis,
        Float64.(SOURCE_COUNTS),
        gpu_peak_gib;
        color=COLORS[GPU_LABEL],
        linestyle=LINESTYLES[GPU_LABEL],
        marker=MARKERS[GPU_LABEL],
        linewidth=1.8,
        markersize=6.5,
    )

    capacity_gib = usable_gpu_bytes / 2.0^30
    hlines!(
        gpu_axis,
        [capacity_gib];
        color=(:black, 0.65),
        linestyle=:dash,
        linewidth=1.0,
    )
    text!(
        gpu_axis,
        900.0,
        capacity_gib;
        text=@sprintf("Usable capacity: %.2f GiB", capacity_gib),
        align=(:left, :top),
        offset=(0, -2),
        fontsize=7.5,
    )
    finite_gpu_peaks = filter(isfinite, gpu_peak_gib)
    gpu_upper = max(capacity_gib, maximum(finite_gpu_peaks)) * 1.08
    ylims!(gpu_axis, 0, gpu_upper)
    xlims!(gpu_axis, 760, 8.0e6)

    gpu_largest = row_for(rows, GPU_LABEL, last(SOURCE_COUNTS))
    if gpu_largest["status"] == "oom"
        add_oom_marker!(
            gpu_axis,
            last(SOURCE_COUNTS),
            min(capacity_gib * 0.92, gpu_upper * 0.9),
        )
    end

    host_values = Float64[]
    for hardware_label in IMPLEMENTATION_LABELS
        append!(
            host_values,
            filter(
                isfinite,
                plot_series!(
                    host_axis,
                    rows,
                    hardware_label,
                    "host_allocated_bytes";
                    label=hardware_label,
                ),
            ),
        )
    end
    host_limits = log_limits(host_values)
    ylims!(host_axis, host_limits...)
    xlims!(host_axis, 760, 8.0e6)

    add_panel_label!(gpu_axis, "(a)")
    add_panel_label!(host_axis, "(b)")
    Legend(
        figure[1, 1:2],
        host_axis;
        orientation=:horizontal,
        nbanks=1,
        framevisible=false,
        labelsize=7.5,
        patchsize=(13, 8),
        colgap=10,
        tellheight=true,
    )
    colgap!(figure.layout, 12)
    rowgap!(figure.layout, 2)
    save_vector_pair(figure, output_dir, "fig4_memory_scaling")
end

function format_table_number(value)
    value isa Number || return "--"
    absolute = abs(Float64(value))
    if absolute != 0 && (absolute < 1.0e-3 || absolute >= 1.0e4)
        return @sprintf("%.3e", value)
    end
    return @sprintf("%.4g", value)
end

latex_escape(value) = replace(string(value), "_" => raw"\_")

function write_table2(path, rows)
    source_count = 1_000_000
    julia = row_for(rows, JULIA_LABEL, source_count)
    cpu = row_for(rows, CPU_LABEL, source_count)
    gpu = row_for(rows, GPU_LABEL, source_count)
    all(valid_runtime, (julia, cpu, gpu)) || error(
        "Table II requires valid 1,000,000-source results for all implementations",
    )

    table_rows = [
        (
            "Julia",
            "CPU, 1 Julia thread",
            "--",
            julia["runtime_seconds"],
            julia["runtime_seconds"],
        ),
        (
            "Reactant CPU",
            "XLA default CPU parallelism",
            cpu["compile_seconds"],
            cpu["runtime_seconds"],
            cpu["runtime_seconds"] + cpu["materialization_seconds"],
        ),
        (
            "Reactant GPU",
            "1x NVIDIA RTX 5090",
            gpu["compile_seconds"],
            gpu["runtime_seconds"],
            gpu["runtime_seconds"] + gpu["materialization_seconds"],
        ),
    ]

    open(path, "w") do io
        println(io, raw"\begin{tabular}{llrrrrrr}")
        println(io, raw"\toprule")
        println(
            io,
            "Implementation & Execution resource & Sources & Compile (s) & Resident (s) & Materialized (s) & Sources/s & Speedup \\\\",
        )
        println(io, raw"\midrule")
        for (implementation, resource, compile, resident, materialized) in
            table_rows
            cells = [
                implementation,
                resource,
                "1,000,000",
                compile isa String ? compile : format_table_number(compile),
                format_table_number(resident),
                format_table_number(materialized),
                format_table_number(source_count / resident),
                format_table_number(julia["runtime_seconds"] / resident),
            ]
            println(io, join(cells, " & "), " \\\\")
        end
        println(io, raw"\bottomrule")
        println(io, raw"\end{tabular}")
    end
    println("wrote $path")
end

function largest_validated_row(rows, hardware_label)
    candidates = [
        row for row in rows if
        row["hardware_label"] == hardware_label &&
        row["status"] == "complete" &&
        row["correctness_passed"] === true
    ]
    isempty(candidates) && error(
        "Table III requires at least one validated $hardware_label result",
    )
    sort!(candidates; by=row -> row["source_count"])
    return last(candidates)
end

function write_table3(path, rows)
    cpu = largest_validated_row(rows, CPU_LABEL)
    gpu = largest_validated_row(rows, GPU_LABEL)
    open(path, "w") do io
        println(io, "% Numerical agreement uses rtol=2e-6, atol=1e-8.")
        println(io, raw"\begin{tabular}{lrrrll}")
        println(io, raw"\toprule")
        println(
            io,
            "Backend & Largest tested sources & Pass & Maximum absolute difference & Maximum relative difference & Worst output field \\\\",
        )
        println(io, raw"\midrule")
        for (label, row) in (("Reactant CPU", cpu), ("Reactant GPU", gpu))
            cells = [
                label,
                replace(string(row["source_count"]), r"(?<=\d)(?=(\d{3})+$)" => ","),
                row["correctness_passed"] === true ? "Yes" : "No",
                format_table_number(row["max_absolute_difference"]),
                format_table_number(row["max_relative_difference"]),
                latex_escape(row["worst_output_field"]),
            ]
            println(io, join(cells, " & "), " \\\\")
        end
        println(io, raw"\bottomrule")
        println(io, raw"\end{tabular}")
    end
    println("wrote $path")
end

function csv_value(value)
    isnothing(value) && return ""
    text = string(value)
    if occursin(',', text) || occursin('"', text) || occursin('\n', text)
        return string('"', replace(text, "\"" => "\"\""), '"')
    end
    return text
end

function write_summaries(
    output_dir,
    results_dir,
    rows,
    usable_gpu_bytes,
    break_evens,
)
    summary = Dict{String,Any}(
        "schema_version" => 1,
        "results_directory" => results_dir,
        "correctness_tolerances" =>
            Dict("rtol" => 2.0e-6, "atol" => 1.0e-8),
        "gpu_usable_memory_bytes" => usable_gpu_bytes,
        "break_even_k" => break_evens,
        "rows" => rows,
    )
    json_path = joinpath(output_dir, "results_summary.json")
    open(json_path, "w") do io
        JSON.json(io, summary; pretty=true)
        println(io)
    end

    csv_path = joinpath(output_dir, "results_summary.csv")
    open(csv_path, "w") do io
        println(io, join(SUMMARY_COLUMNS, ","))
        for row in rows
            println(
                io,
                join((csv_value(get(row, column, nothing)) for
                      column in SUMMARY_COLUMNS), ","),
            )
        end
    end
    println("wrote $json_path")
    println("wrote $csv_path")
end

function format_gib(bytes)
    bytes isa Number || return "unavailable"
    return @sprintf("%.2f GiB", bytes / 2.0^30)
end

function metadata_versions(invocations, key)
    values = sort(unique(string(get(invocation, key, "unknown")) for
                         invocation in invocations))
    return join(values, ", ")
end

function hardware_value(invocations, key)
    values = Any[]
    for invocation in invocations
        hardware = get(invocation, "hardware", Dict{String,Any}())
        value = get(hardware, key, nothing)
        isnothing(value) || push!(values, value)
    end
    isempty(values) && return nothing
    unique_values = unique(values)
    return length(unique_values) == 1 ? first(unique_values) :
           join(string.(unique_values), ", ")
end

function format_source_count(source_count)
    return replace(string(source_count), r"(?<=\d)(?=(\d{3})+$)" => ",")
end

function write_handoff(
    path,
    metadata_document,
    rows,
    usable_gpu_bytes,
    break_evens,
)
    invocations = sort(
        metadata_document["invocations"];
        by=entry -> (entry["source_count"], entry["selected_backend"]),
    )
    included = [
        invocation for invocation in invocations if
        invocation["source_count"] in SOURCE_COUNTS &&
        invocation["selected_backend"] in ("CPU", "CUDA")
    ]
    isempty(included) && error("metadata contains no paper benchmark invocations")

    validated = [
        row for row in rows if
        row["correctness_passed"] === true && row["status"] == "complete"
    ]
    worst_absolute = isempty(validated) ? nothing :
                     maximum(row["max_absolute_difference"] for row in validated)
    worst_relative_row = isempty(validated) ? nothing :
                         argmax(
        row -> row["max_relative_difference"],
        validated,
    )
    problem_rows = [
        row for row in rows if row["status"] != "complete"
    ]

    open(path, "w") do io
        println(io, "# SkyMatch.jl benchmark handoff")
        println(io)
        println(
            io,
            "This directory is generated entirely from the direct JSON files in " *
            "benchmark/results/. Figure 1 is not generated here; it is authored " *
            "in the manuscript's LaTeX/TikZ source.",
        )
        println(io)
        println(io, "## Included invocations")
        println(io)
        println(io, "| Backend invocation | Sources | Commit | Dirty | Status |")
        println(io, "|---|---:|---|:---:|---|")
        for invocation in included
            println(
                io,
                "| $(invocation["selected_backend"]) | " *
                "$(format_source_count(invocation["source_count"])) | " *
                "$(invocation["source_commit"]) | " *
                "$(invocation["dirty"]) | $(invocation["status"]) |",
            )
        end
        println(io)

        commits = unique(string(entry["source_commit"]) for entry in included)
        if length(commits) > 1
            println(
                io,
                "**Revision mismatch:** included raw results span commits " *
                "$(join(commits, ", ")). Treat cross-revision comparisons with caution.",
            )
        elseif any(entry["dirty"] === true for entry in included)
            println(
                io,
                "All included invocations name commit $(first(commits)), but the " *
                "working tree was dirty for at least one invocation; the dirty " *
                "status is retained above rather than presenting the data as a " *
                "clean-revision result.",
            )
        else
            println(io, "All included invocations use clean commit $(first(commits)).")
        end
        println(io)

        println(io, "## Hardware and software")
        println(io)
        println(io, "- Julia: $(metadata_versions(included, "julia_version"))")
        println(io, "- Reactant: $(metadata_versions(included, "reactant_version"))")
        println(
            io,
            "- CUDA runtime: $(something(hardware_value(included, "cuda_runtime_version"), "unavailable"))",
        )
        println(
            io,
            "- cuDNN: $(something(hardware_value(included, "cudnn_version"), "unavailable"))",
        )
        println(
            io,
            "- NVIDIA driver: $(something(hardware_value(included, "nvidia_driver_version"), "unavailable"))",
        )
        println(io, "- OS: $(something(hardware_value(included, "os"), "unavailable"))")
        println(
            io,
            "- Kernel: $(something(hardware_value(included, "kernel"), "unavailable"))",
        )
        println(
            io,
            "- CPU: $(something(hardware_value(included, "cpu_model"), "unavailable"))",
        )
        println(
            io,
            "- Logical CPU threads: $(something(hardware_value(included, "logical_cpu_threads"), "unavailable"))",
        )
        println(
            io,
            "- RAM: $(format_gib(hardware_value(included, "ram_bytes")))",
        )
        println(
            io,
            "- GPU: $(something(hardware_value(included, "gpu_name"), "unavailable"))",
        )
        println(
            io,
            "- Selected GPU index: $(something(hardware_value(included, "selected_gpu_index"), "unavailable"))",
        )
        println(
            io,
            "- Hardware-reported usable GPU memory: $(format_gib(usable_gpu_bytes)) " *
            "($(usable_gpu_bytes) bytes)",
        )
        println(io)

        println(io, "## Timing and sampling definitions")
        println(io)
        println(
            io,
            "- Resident execution is a synchronized call on already placed arrays " *
            "with an already compiled executable. It excludes loading, preparation, " *
            "placement, compilation, host materialization, DataFrame assembly, and FITS writing.",
        )
        println(
            io,
            "- Materialized time is resident execution plus conversion of every " *
            "returned array to host arrays, with host completion forced.",
        )
        println(
            io,
            "- One-shot time is the sum of measured input placement, measured " *
            "compilation, the first synchronized execution, and complete output materialization.",
        )
        println(
            io,
            "- Julia and Reactant CPU use the scalar Chairmarks bench.time with a " *
            "five-second budget, one evaluation per sample, and 100 requested samples.",
        )
        println(
            io,
            "- Reactant GPU uses the synchronized XProf mean across 25 executions. " *
            "Its host-wrapper allocation measurement uses one Chairmarks sample; " *
            "XProf FLOP/s remains raw data and is not a headline result.",
        )
        println(
            io,
            "- Julia CPU uses one Julia thread. Reactant CPU uses XLA's default CPU " *
            "parallelization and is not a single-core measurement.",
        )
        println(
            io,
            "- Filters are disabled and explicit noise uses seed " *
            "$(metadata_versions(included, "seed")).",
        )
        println(io)

        println(io, "## Numerical agreement")
        println(io)
        println(
            io,
            "Every valid backend result is checked field by field against ordinary " *
            "Julia with rtol=2e-6 and atol=1e-8. Relative differences use " *
            "max(abs(reference), atol) as the denominator.",
        )
        if !isnothing(worst_absolute)
            println(
                io,
                "- Worst observed absolute difference: " *
                "$(format_table_number(worst_absolute)).",
            )
            println(
                io,
                "- Worst observed relative difference: " *
                "$(format_table_number(worst_relative_row["max_relative_difference"])) " *
                "in $(worst_relative_row["worst_output_field"]) on " *
                "$(worst_relative_row["hardware_label"]) at " *
                "$(format_source_count(worst_relative_row["source_count"])) sources.",
            )
        else
            println(io, "- No successfully validated Reactant result is available.")
        end
        println(io)

        println(io, "## OOM and missing cases")
        println(io)
        if isempty(problem_rows)
            println(io, "- None.")
        else
            for row in problem_rows
                println(
                    io,
                    "- $(row["hardware_label"]), " *
                    "$(format_source_count(row["source_count"])) sources: " *
                    "$(row["status"]).",
                )
            end
        end
        println(io)

        println(io, "## Compilation amortization")
        println(io)
        println(
            io,
            "The Figure 3 curves are calculated from the measured placement, " *
            "compilation, resident-execution, and materialization components; " *
            "they are not separately timed experiments.",
        )
        for result in break_evens
            break_even = result["break_even_k"]
            description = isnothing(break_even) ?
                          "no break-even through K=100,000" :
                          "K=$(break_even)"
            println(
                io,
                "- $(result["backend"]), $(result["workflow"]) workflow: $description.",
            )
        end
        println(io)

        println(io, "## Claims supported by these data")
        println(io)
        all(row["correctness_passed"] === true for row in validated) &&
            println(
                io,
                "- The successfully measured Reactant points agree with ordinary " *
                "Julia at the stated tolerances.",
            )
        cpu_small = row_for(rows, CPU_LABEL, first(SOURCE_COUNTS))
        julia_small = row_for(rows, JULIA_LABEL, first(SOURCE_COUNTS))
        later_crossovers = [
            source_count for source_count in SOURCE_COUNTS[2:end] if
            valid_runtime(row_for(rows, CPU_LABEL, source_count)) &&
            valid_runtime(row_for(rows, JULIA_LABEL, source_count)) &&
            row_for(rows, JULIA_LABEL, source_count)["runtime_seconds"] >
            row_for(rows, CPU_LABEL, source_count)["runtime_seconds"]
        ]
        if valid_runtime(cpu_small) && valid_runtime(julia_small) &&
           cpu_small["runtime_seconds"] >= julia_small["runtime_seconds"] &&
           !isempty(later_crossovers)
            println(
                io,
                "- Reactant CPU crosses from no warm-speed advantage at the " *
                "smallest catalog to a warm-speed advantage by " *
                "$(format_source_count(first(later_crossovers))) sources.",
            )
        end
        gpu_faster_points = [
            source_count for source_count in SOURCE_COUNTS if
            valid_runtime(row_for(rows, GPU_LABEL, source_count)) &&
            valid_runtime(row_for(rows, CPU_LABEL, source_count)) &&
            row_for(rows, GPU_LABEL, source_count)["runtime_seconds"] <
            row_for(rows, CPU_LABEL, source_count)["runtime_seconds"]
        ]
        !isempty(gpu_faster_points) && println(
            io,
            "- The single RTX 5090 has lower warm resident time than Reactant CPU " *
            "at the measured points: " *
            "$(join(format_source_count.(gpu_faster_points), ", ")).",
        )
        println(
            io,
            "- The data support measured warm-time, speedup, overhead-component, " *
            "GPU allocator-peak, and host-allocation comparisons at the recorded sizes.",
        )
        println(io)

        println(io, "## Claims not supported by these data")
        println(io)
        println(io, "- Uncertainty intervals or process-to-process variability.")
        println(io, "- Single-core Reactant CPU performance.")
        println(io, "- End-to-end CSV/FITS pipeline latency or peak host resident memory.")
        println(io, "- Filter-enabled performance.")
        println(io, "- TPU performance.")
        println(io, "- An extrapolated runtime or memory value for an OOM/missing point.")
        println(io)

        println(io, "Regenerate all artifacts with:")
        println(io)
        println(io, "    julia --startup-file=no --history-file=no --project=benchmark \\")
        println(io, "      benchmark/plots/plot_results.jl \\")
        println(io, "      --results-dir benchmark/results \\")
        println(io, "      --output-dir benchmark/plots/generated")
    end
    println("wrote $path")
end

function main(args=ARGS)
    options = parse_options(args)
    results_dir = options["--results-dir"]
    output_dir = options["--output-dir"]
    isdir(results_dir) || error("results directory does not exist: $results_dir")
    mkpath(output_dir)

    metadata_document, metadata_index = read_metadata(results_dir)
    rows = build_rows(results_dir, metadata_index)
    any(row["filters_enabled"] === true for row in rows) && error(
        "paper plots require filters_enabled=false for every included result",
    )
    usable_gpu_bytes = gpu_capacity_bytes(metadata_document)
    break_evens = calculate_break_evens(rows)

    make_figure2(output_dir, rows)
    make_figure3(output_dir, rows, break_evens)
    make_figure4(output_dir, rows, usable_gpu_bytes)
    write_table2(joinpath(output_dir, "table2_largest_common.tex"), rows)
    write_table3(joinpath(output_dir, "table3_correctness.tex"), rows)
    write_summaries(
        output_dir,
        results_dir,
        rows,
        usable_gpu_bytes,
        break_evens,
    )
    write_handoff(
        joinpath(output_dir, "BENCHMARK_HANDOFF.md"),
        metadata_document,
        rows,
        usable_gpu_bytes,
        break_evens,
    )
    return nothing
end

main()
