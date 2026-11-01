using Statistics: quantile

function lir_plot_source(results_dir)
    # Adding larger runtime measurements must not change the agreed histogram.
    n = 1_000_000
    stem = joinpath(results_dir, "bayesmmfwd_lir_sensitivity_$(n)_CPUbenchmarks")
    all(isfile(stem * suffix) for suffix in ("_per_galaxy.json", "_correctness.json", "_metadata.json")) ||
        return nothing
    get(JSON.parsefile(stem * "_correctness.json"), "passed", false) === true || return nothing
    get(JSON.parsefile(stem * "_metadata.json"), "status", "missing") == "complete" || return nothing
    return n
end

function lir_histogram(values, log_edges)
    counts = zeros(Int, length(log_edges) - 1)
    for value in values
        value > 0 || continue
        bin = clamp(searchsortedlast(log_edges, log10(value)), 1, length(counts))
        counts[bin] += 1
    end
    return counts
end

function lir_measurements(results_dir, n)
    cpu_stem = joinpath(results_dir, "bayesmmfwd_lir_sensitivity_$(n)_CPUbenchmarks")
    for suffix in (".json", "_metadata.json", "_correctness.json")
        isfile(cpu_stem * suffix) || error(
            "Missing LIR sensitivity result: $(cpu_stem * suffix). Run " *
            "benchmark/benchmark_lir_sensitivity.jl --rows $n --backend CPU --results-dir $results_dir.",
        )
    end
    cpu_metadata = JSON.parsefile(cpu_stem * "_metadata.json")
    cpu_checks = JSON.parsefile(cpu_stem * "_correctness.json")
    cpu_metadata["status"] == "complete" && cpu_checks["passed"] === true ||
        error("invalid CPU LIR sensitivity measurement at $n galaxies")
    cpu_runtime = JSON.parsefile(cpu_stem * ".json")
    measurement_rows = Dict{String,Any}[
        Dict{String,Any}(
            "hardware_label" => JULIA_LABEL,
            "source_count" => n,
            "runtime_seconds" => benchmark_value(cpu_runtime, "CPU", "Julia"),
            "comparison" => "five-point finite difference at h/2",
            "correctness" => cpu_checks["julia_vs_finite_difference_half_h"],
            "metadata" => cpu_metadata,
        ),
        Dict{String,Any}(
            "hardware_label" => CPU_LABEL,
            "source_count" => n,
            "runtime_seconds" => benchmark_value(cpu_runtime, "CPU", "Default"),
            "comparison" => "Julia forward-mode Enzyme",
            "correctness" => cpu_checks["reactant_vs_julia"],
            "metadata" => cpu_metadata,
        ),
    ]
    gpu_stem = joinpath(results_dir, "bayesmmfwd_lir_sensitivity_$(n)_CUDAbenchmarks")
    gpu_metadata_path = gpu_stem * "_metadata.json"
    gpu_metadata = isfile(gpu_metadata_path) ? JSON.parsefile(gpu_metadata_path) : nothing
    gpu_status = isnothing(gpu_metadata) ? "missing" : gpu_metadata["status"]
    gpu_status == "complete" || error(
        "LIR sensitivity runtime panel needs validated CUDA data for $n galaxies (status=$gpu_status). " *
        "Run benchmark/benchmark_lir_sensitivity.jl --rows $n --backend CUDA --results-dir $results_dir.",
    )
    gpu_checks = JSON.parsefile(gpu_stem * "_correctness.json")
    get(gpu_checks, "passed", false) === true || error("invalid GPU LIR correctness")
    gpu_runtime = JSON.parsefile(gpu_stem * ".json")
    for key in ("source_count", "seed", "dataset", "nominal_sigma_ms_dex", "sfr2lir", "source_commit")
        gpu_metadata[key] == cpu_metadata[key] || error("CPU/GPU LIR metadata mismatch: $key")
    end
    push!(measurement_rows, Dict{String,Any}(
        "hardware_label" => GPU_LABEL,
        "source_count" => n,
        "runtime_seconds" => benchmark_value(gpu_runtime, "CUDA", "Default"),
        "comparison" => "Julia forward-mode Enzyme",
        "correctness" => gpu_checks["reactant_vs_julia"],
        "metadata" => gpu_metadata,
    ))
    for row in measurement_rows
        runtime = row["runtime_seconds"]
        runtime isa Real && isfinite(runtime) && runtime > 0 ||
            error("invalid LIR sensitivity runtime for $(row["hardware_label"])")
        row["correctness"]["passed"] === true ||
            error("invalid LIR sensitivity derivative for $(row["hardware_label"])")
        row["speedup_over_julia_cpu"] = first(measurement_rows)["runtime_seconds"] / runtime
        row["status"] = "complete"
    end
    return measurement_rows, gpu_status
end

function make_lir_sensitivity_artifacts(output_dir, results_dir; required=false)
    n = lir_plot_source(results_dir)
    if isnothing(n)
        required && error(
            "No validated LIR sensitivity data. Run benchmark/benchmark_lir_sensitivity.jl " *
            "--rows 1000000 --backend CPU --results-dir benchmark/results first.",
        )
        return nothing
    end
    cpu_stem = joinpath(results_dir, "bayesmmfwd_lir_sensitivity_$(n)_CPUbenchmarks")
    data = JSON.parsefile(cpu_stem * "_per_galaxy.json")
    cpu_metadata = JSON.parsefile(cpu_stem * "_metadata.json")
    cpu_checks = JSON.parsefile(cpu_stem * "_correctness.json")
    measurement_rows = Dict{String,Any}[]
    for source_count in SOURCE_COUNTS
        rows, _ = lir_measurements(results_dir, source_count)
        for row in rows, key in (
            "seed", "dataset", "nominal_sigma_ms_dex", "sfr2lir", "source_commit",
            "ad_mode", "julia_version", "reactant_version", "enzyme_version",
        )
            row["metadata"][key] == cpu_metadata[key] ||
                error("LIR scaling metadata mismatch at $source_count: $key")
        end
        append!(measurement_rows, rows)
    end
    gpu_status = "complete"
    lir = Float64.(data["nominal_lir_lsun"])
    derivative = Float64.(data["julia_d_lir_d_sigma_lsun_per_dex"])
    passive = Bool.(data["qflag"])
    starburst = Bool.(data["issb"])
    all(length(x) == n for x in (lir, derivative, passive, starburst)) || error("LIR record length mismatch")
    all(isfinite, lir) && all(isfinite, derivative) || error("non-finite LIR data")
    active_indices = findall(>(0), lir)
    isempty(active_indices) && error("no positive LIR values to plot")
    increasing = derivative[derivative .> 0]
    decreasing = -derivative[derivative .< 0]
    nonzero = abs.(derivative[derivative .!= 0])
    isempty(nonzero) && error("all LIR derivatives are zero")
    zero_count = count(iszero, derivative)

    # Compare the same warm forward-mode derivative calculation, not the
    # primal model or the older a0 sensitivity experiment.
    runtimes = [row["runtime_seconds"] for row in measurement_rows]
    runtime_limits = (minimum(runtimes) / 2, maximum(runtimes) * 4)
    runtime_exponents = ceil(Int, log10(runtime_limits[1])):floor(Int, log10(runtime_limits[2]))
    runtime_ticks = (10.0 .^ runtime_exponents,
        [rich("10", superscript(string(exponent))) for exponent in runtime_exponents])
    maximum_magnitude = maximum(nonzero)
    sign_colors = (COLORS[CPU_LABEL], COLORS[GPU_LABEL])
    figure = Figure(size=(515, 216), fontsize=8)
    runtime_axis = Axis(
        figure[2, 1]; AXIS_STYLE...,
        xlabel="Number of galaxies, N_src",
        ylabel="Warm forward-mode time (s)",
        xscale=log10,
        xticks=(SOURCE_COUNTS, SOURCE_TICK_LABELS),
        yscale=log10,
        yticks=runtime_ticks,
    )
    distribution_axis = Axis(
        figure[2, 2]; AXIS_STYLE...,
        xlabel="|dLIR / dσ_MS| (L☉ dex⁻¹)",
        ylabel="Galaxies per log-spaced bin",
        xscale=log10,
        yscale=log10,
    )
    for label in IMPLEMENTATION_LABELS
        plot_series!(runtime_axis, measurement_rows, label, "runtime_seconds"; label)
    end
    xlims!(runtime_axis, first(SOURCE_COUNTS) / 1.4, last(SOURCE_COUNTS) * 1.4)
    ylims!(runtime_axis, runtime_limits...)
    low = floor(log10(minimum(nonzero)))
    high = ceil(log10(maximum_magnitude))
    high <= low && (high = low + 1)
    log_edges = collect(range(low, high; length=61))
    edges = 10.0 .^ log_edges
    histograms = (lir_histogram(increasing, log_edges), lir_histogram(decreasing, log_edges))
    for (counts, color, linestyle, name) in zip(
        histograms, sign_colors, (:solid, :dash), ("Increasing LIR", "Decreasing LIR"),
    )
        values = [count > 0 ? Float64(count) : NaN for count in counts]
        stairs!(distribution_axis, edges, [values; last(values)];
            step=:post, color, linestyle, linewidth=1.6, label=name)
    end
    xlims!(distribution_axis, first(edges), last(edges))
    ylims!(distribution_axis, 0.8, 10.0maximum(maximum.(histograms)))
    text!(distribution_axis, 0.98, 0.98; space=:relative,
        text="$(format_source_count(zero_count)) zero derivatives\n($(format_source_count(count(passive))) passive galaxies)",
        align=(:right, :top), fontsize=7.5)
    add_panel_label!(runtime_axis, "(a)")
    add_panel_label!(distribution_axis, "(b)")
    Legend(figure[1, 1], runtime_axis; orientation=:vertical,
        framevisible=false, labelsize=7.5, patchsize=(20, 8), rowgap=1,
        tellheight=true, tellwidth=false)
    # Separate legends: backend on the left, derivative sign on the right.
    Legend(figure[1, 2], distribution_axis, "$(format_source_count(n)) galaxies";
        orientation=:horizontal, framevisible=false, labelsize=8, patchsize=(20, 8),
        titlesize=8, titlefont=:regular, titlegap=3, tellheight=true, tellwidth=false)
    colgap!(figure.layout, 12)
    rowgap!(figure.layout, 2)
    save_vector_pair(figure, output_dir, "lir_sigma_ms_sensitivity")

    summary = Dict{String,Any}(
        "schema_version" => 3,
        "source_count" => n,
        "histogram_source_count" => n,
        "runtime_source_counts" => SOURCE_COUNTS,
        "metric" => "per-galaxy dLIR_i/dsigma_MS",
        "left_panel_metric" => "warm_forward_mode_runtime_seconds",
        "nominal_sigma_ms_dex" => data["nominal_sigma_ms_dex"],
        "units" => "L_sun/dex",
        "passive_galaxies" => count(passive),
        "starburst_galaxies" => count(starburst),
        "increasing_lir_galaxies" => length(increasing),
        "decreasing_lir_galaxies" => length(decreasing),
        "zero_derivative_galaxies" => zero_count,
        "derivative_min_lsun_per_dex" => minimum(derivative),
        "derivative_max_lsun_per_dex" => maximum(derivative),
        "derivative_quantile_probabilities" => [0.01, 0.1, 0.5, 0.9, 0.99],
        "active_derivative_quantiles_lsun_per_dex" => quantile(derivative[active_indices], [0.01, 0.1, 0.5, 0.9, 0.99]),
        "speedup_baseline" => "Julia CPU (1 thread), warm forward-mode per-galaxy dLIR_i/dsigma_MS",
        "histogram_uses_all_galaxies" => true,
        "histogram_log10_edges" => log_edges,
        "increasing_lir_histogram_counts" => histograms[1],
        "decreasing_lir_histogram_counts" => histograms[2],
        "finite_difference_steps_dex" => cpu_checks["finite_difference_steps_dex"],
        "finite_difference_step_stability" => cpu_checks["finite_difference_step_stability"],
        "gpu_status" => gpu_status,
        "measurements" => measurement_rows,
    )
    open(joinpath(output_dir, "lir_sigma_ms_sensitivity_summary.json"), "w") do io
        JSON.json(io, summary; pretty=true)
        println(io)
    end
    open(joinpath(output_dir, "lir_sigma_ms_sensitivity_summary.csv"), "w") do io
        println(io, "implementation,source_count,runtime_seconds,speedup_over_julia_cpu,reference,passed,max_absolute_difference_lsun_per_dex,max_difference_divided_by_nominal_lir")
        for row in measurement_rows
            check = row["correctness"]
            values = [row["hardware_label"], row["source_count"], row["runtime_seconds"], row["speedup_over_julia_cpu"], row["comparison"], check["passed"],
                check["max_absolute_difference_lsun_per_dex"], check["max_difference_divided_by_nominal_lir"]]
            println(io, join(csv_value.(values), ","))
        end
    end
    open(joinpath(output_dir, "LIR_SENSITIVITY_README.md"), "w") do io
        println(io, "# Per-galaxy infrared-luminosity sensitivity\n")
        println(io, "Runtime scaling uses $(join(format_source_count.(SOURCE_COUNTS), "; ")) galaxies (the first N catalog rows), at sigma_MS = $(data["nominal_sigma_ms_dex"]) dex with seed $(data["seed"]). The sensitivity histogram remains fixed at $(format_source_count(n)) galaxies.\n")
        println(io, "The unchanged SFR kernel is followed by LIR_i = SFR2LIR * SFR_i, the same conversion used by generate_dust_outputs. One forward-mode Enzyme pass returns one derivative per galaxy. Timings are for this SFR-to-LIR dependency, not the full multiwavelength/line simulation. All other parameters and random draws are held fixed.\n")
        println(io, "sigma_MS is shared by main-sequence and starburst scatter in the existing model. Galaxy classification stays fixed. Units: L_sun per dex of scatter. This is sensitivity of each realized galaxy, not an expectation over fresh random draws.\n")
        println(io, "Panel (a) shows the recorded warm forward-mode derivative-execution times for Julia CPU (1 thread), Reactant CPU (XLA default), and Reactant GPU (1x RTX 5090), at each catalog size. Each point is one recorded measurement using the harness's internal sampling, without external replication or error bars. Compilation, input placement, and output materialization are excluded. These are forward-mode timings for one parameter and N per-galaxy derivatives, not reverse-mode timings from the older scalar SFR experiment or timings for the full simulation.\n")
        println(io, "Panel (b) is unchanged: all positive and negative derivatives are counted in the same log-spaced magnitude bins, with zero derivatives counted separately. Lines are histogram outlines, not fits or uncertainty bounds. Its legend denotes the derivative sign, not the execution backend.\n")
        println(io, "Increasing / decreasing / zero derivatives: $(length(increasing)) / $(length(decreasing)) / $zero_count. Passive galaxies: $(count(passive)); starbursts: $(count(starburst)).\n")
        println(io, "Every CPU derivative was checked against five-point finite differences at h=$(cpu_checks["finite_difference_steps_dex"][1]) and h/2=$(cpu_checks["finite_difference_steps_dex"][2]) dex. Both reference checks and step stability must pass the recorded per-galaxy scaled tolerance. Every available Reactant derivative is checked against Julia. Full arrays, including passive galaxies, remain in the raw per_galaxy.json files.\n")
        println(io, "The per-galaxy acceptance rule is abs(candidate - reference) <= rtol * abs(reference) + scaled_atol * max(abs(nominal LIR), 1 L_sun). Finite differences use rtol=2e-5 and scaled_atol=1e-7; backend agreement uses rtol=2e-6 and scaled_atol=1e-8. The absolute term scales with luminosity; it is not a fixed 1e-8 L_sun tolerance.\n")
        println(io, "Julia and Reactant CPU use Chairmarks (5 s budget, 100 requested samples, one evaluation/sample); GPU uses the existing synchronized 25-execution XProf mean. GPU status: $gpu_status.\n")
        for row in measurement_rows
            metadata = row["metadata"]
            println(io, "- $(row["hardware_label"]), $(format_source_count(row["source_count"])) galaxies: $(row["runtime_seconds"]) s; $(row["speedup_over_julia_cpu"])x over Julia CPU at the same size; commit $(metadata["source_commit"]), dirty=$(metadata["dirty"]); Julia $(metadata["julia_version"]), Enzyme $(metadata["enzyme_version"]), Reactant $(metadata["reactant_version"]).")
        end
        println(io, "\n## Reproduce\n")
        println(io, "Run from the BayesMMfwd.jl root. Each benchmark command is needed once; reuse existing validated results. The plotting command does not rerun measurements.\n\n```bash")
        for backend in ("CPU", "CUDA"), source_count in SOURCE_COUNTS
            println(io, "julia --startup-file=no --history-file=no --project=benchmark benchmark/benchmark_lir_sensitivity.jl --rows $source_count --backend $backend --results-dir benchmark/results")
        end
        println(io, "julia --startup-file=no --history-file=no --project=benchmark benchmark/plots/plot_results.jl --results-dir benchmark/results --output-dir benchmark/plots/generated --only lir-sensitivity")
        println(io, "```\n\nThe canonical plot command without --only also includes these artifacts when validated LIR data exist. Paper source and existing SFR artifacts have not been replaced.")
    end
    println("wrote LIR sensitivity summaries and reproduction notes")
    return summary
end
