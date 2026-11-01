function read_pysides_results(results_dir)
    rows = Dict{Int,Dict{String,Any}}()
    for source_count in SOURCE_COUNTS
        path = joinpath(
            results_dir,
            "pysides_forward_$(source_count)_CPUbenchmarks.json",
        )
        if !isfile(path)
            rows[source_count] = Dict{String,Any}(
                "source_count" => source_count,
                "status" => "missing",
                "runtime_seconds" => nothing,
            )
            continue
        end
        row = Dict{String,Any}(JSON.parsefile(path))
        Int(row["source_count"]) == source_count || error(
            "$path records source_count=$(row["source_count"]), expected $source_count",
        )
        get(row, "filters_enabled", nothing) === false || error(
            "$path must record filters_enabled=false",
        )
        get(row, "seed", nothing) == 1234 || error(
            "$path must record seed=1234",
        )
        rows[source_count] = row
    end
    return rows
end

function make_figure5(output_dir, rows, pysides_rows)
    figure = Figure(size=(515, 210), fontsize=8)
    axis = Axis(
        figure[2, 1];
        AXIS_STYLE...,
        xscale=log10,
        yscale=log10,
        xlabel="Number of galaxies, N_src",
        ylabel="Model execution time (s)",
        xticks=(Float64.(SOURCE_COUNTS), SOURCE_TICK_LABELS),
    )


    runtime_values = Float64[]
    pysides_values = Float64[]
    for source_count in SOURCE_COUNTS
        row = pysides_rows[source_count]
        value = get(row, "runtime_seconds", nothing)
        valid = get(row, "status", "missing") == "complete" &&
                value isa Number && value > 0
        plotted = valid ? Float64(value) : NaN
        push!(pysides_values, plotted)
        isfinite(plotted) && push!(runtime_values, plotted)
    end
    any(isfinite, pysides_values) || error(
        "Figure 5 requires at least one complete pySIDES benchmark result",
    )
    scatterlines!(
        axis,
        Float64.(SOURCE_COUNTS),
        pysides_values;
        color=COLORS[PYSIDES_LABEL],
        linestyle=LINESTYLES[PYSIDES_LABEL],
        marker=MARKERS[PYSIDES_LABEL],
        linewidth=1.8,
        markersize=6.5,
        label=PYSIDES_LABEL,
    )

    for hardware_label in (CPU_LABEL, JULIA_LABEL, GPU_LABEL)
        values = Float64[]
        for source_count in SOURCE_COUNTS
            row = row_for(rows, hardware_label, source_count)
            value = valid_runtime(row) ? Float64(row["runtime_seconds"]) : NaN
            push!(values, value)
            isfinite(value) && push!(runtime_values, value)
        end
        scatterlines!(
            axis,
            Float64.(SOURCE_COUNTS),
            values;
            color=COLORS[hardware_label],
            linestyle=LINESTYLES[hardware_label],
            marker=MARKERS[hardware_label],
            linewidth=1.8,
            markersize=6.5,
            label=hardware_label,
        )
    end

    runtime_limits = log_limits(runtime_values; padding=0.15)
    ylims!(axis, runtime_limits...)
    xlims!(axis, 760, 8.0e6)
    failure_y = 10.0^(
        log10(runtime_limits[1]) +
        0.92 * (log10(runtime_limits[2]) - log10(runtime_limits[1]))
    )
    for source_count in SOURCE_COUNTS
        status = String(get(pysides_rows[source_count], "status", "missing"))
        status == "complete" && continue
        scatter!(
            axis,
            [Float64(source_count)],
            [failure_y];
            color=COLORS[PYSIDES_LABEL],
            marker=:xcross,
            markersize=8,
            strokewidth=1.8,
        )
        text!(
            axis,
            Float64(source_count),
            failure_y;
            text=uppercase(status),
            color=COLORS[PYSIDES_LABEL],
            align=(:right, :bottom),
            offset=(-2, 4),
            fontsize=7.5,
            font=:bold,
        )
    end

    Legend(
        figure[1, 1],
        axis;
        orientation=:horizontal,
        nbanks=2,
        framevisible=false,
        labelsize=7.5,
        patchsize=(13, 8),
        colgap=14,
        rowgap=1,
        tellheight=true,
    )
    rowgap!(figure.layout, 1)
    save_vector_pair(figure, output_dir, "fig5_implementation_runtime")
end
function sensitivity_runtime(
    results_dir,
    source_count,
    backend,
    configuration,
)
    path = joinpath(
        results_dir,
        "bayesmmfwd_sfr_sensitivity_$(source_count)_$(backend)benchmarks.json",
    )
    isfile(path) || return nothing
    return benchmark_value(JSON.parsefile(path), backend, configuration)
end

function make_figure7(output_dir, results_dir)
    figure = Figure(size=(515, 190), fontsize=8)
    axis = Axis(
        figure[2, 1];
        AXIS_STYLE...,
        xscale=log10,
        yscale=log10,
        xlabel="Number of galaxies, N_src",
        ylabel="Warm SFR sensitivity time (s)",
        xticks=(Float64.(SOURCE_COUNTS), SOURCE_TICK_LABELS),
    )

    series = (
        (JULIA_LABEL, "CPU", "Julia"),
        (CPU_LABEL, "CPU", "Default"),
        (GPU_LABEL, "CUDA", "Default"),
    )
    all_values = Float64[]
    for (hardware_label, backend, configuration) in series
        values = Float64[]
        for source_count in SOURCE_COUNTS
            value = sensitivity_runtime(
                results_dir,
                source_count,
                backend,
                configuration,
            )
            plotted = value isa Number && value > 0 ? Float64(value) : NaN
            push!(values, plotted)
            isfinite(plotted) && push!(all_values, plotted)
        end
        scatterlines!(
            axis,
            Float64.(SOURCE_COUNTS),
            values;
            color=COLORS[hardware_label],
            linestyle=LINESTYLES[hardware_label],
            marker=MARKERS[hardware_label],
            linewidth=1.8,
            markersize=6.5,
            label=hardware_label,
        )
    end
    isempty(all_values) && error(
        "Figure 7 requires at least one SFR sensitivity benchmark",
    )
    xlims!(axis, 760, 8.0e6)
    ylims!(axis, log_limits(all_values; padding=0.16)...)
    Legend(
        figure[1, 1],
        axis;
        orientation=:horizontal,
        nbanks=1,
        framevisible=false,
        labelsize=7.5,
        patchsize=(13, 8),
        colgap=12,
        tellheight=true,
    )
    rowgap!(figure.layout, 1)
    save_vector_pair(figure, output_dir, "fig7_sfr_sensitivity_runtime")
end

function sensitivity_metrics(results_dir, source_count, backend)
    path = joinpath(
        results_dir,
        "bayesmmfwd_sfr_sensitivity_$(source_count)_$(backend)" *
        "benchmarks_correctness.json",
    )
    isfile(path) || error("missing sensitivity correctness data: $path")
    values = Dict{String,Any}()
    for row in JSON.parsefile(path)
        values[String(row["metric"])] = row["value"]
    end
    get(values, "Pass", false) === true || error(
        "sensitivity correctness did not pass in $path",
    )
    return values
end

function required_sensitivity_metric(metrics, name)
    haskey(metrics, name) || error("missing sensitivity metric: $name")
    value = metrics[name]
    value isa Number || error("sensitivity metric $name is not numeric")
    isfinite(value) || error("sensitivity metric $name is not finite")
    return Float64(value)
end

function write_table4_sensitivity(path, results_dir)
    source_count = 1_000_000
    cpu = sensitivity_metrics(results_dir, source_count, "CPU")
    gpu = sensitivity_metrics(results_dir, source_count, "CUDA")
    finite_difference = required_sensitivity_metric(
        cpu,
        "Finite-difference sensitivity",
    )
    julia_sensitivity = required_sensitivity_metric(
        cpu,
        "Julia Enzyme sensitivity",
    )
    cpu_sensitivity = required_sensitivity_metric(
        cpu,
        "Reactant Enzyme sensitivity",
    )
    gpu_sensitivity = required_sensitivity_metric(
        gpu,
        "Reactant Enzyme sensitivity",
    )
    denominator = max(abs(finite_difference), 1.0e-10)
    methods = [
        ("Five-point finite difference", finite_difference),
        ("Julia CPU + Enzyme", julia_sensitivity),
        ("Reactant CPU + Enzyme", cpu_sensitivity),
        ("Reactant GPU + Enzyme", gpu_sensitivity),
    ]

    open(path, "w") do io
        println(
            io,
            "% Sensitivity is d mean(log1p(SFR / sfr_scale)) / d a0 " *
            "at 1,000,000 sources; finite-difference step=1e-4.",
        )
        println(io, raw"\begin{tabular}{lrrr}")
        println(io, raw"\toprule")
        println(
            io,
            "Method & Sensitivity & Absolute difference & Relative difference \\\\",
        )
        println(io, raw"\midrule")
        for (method, sensitivity) in methods
            absolute = abs(sensitivity - finite_difference)
            relative = absolute / denominator
            cells = [
                method,
                @sprintf("%.14g", sensitivity),
                format_table_number(absolute),
                format_table_number(relative),
            ]
            println(io, join(cells, " & "), " \\\\")
        end
        println(io, raw"\bottomrule")
        println(io, raw"\end{tabular}")
    end
    println("wrote $path")
end
