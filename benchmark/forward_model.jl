using BayesMMfwd: BayesMMfwd
using Reactant: Reactant

include("common.jl")

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

function check_forward_output(reference, candidate; rtol=2.0e-6, atol=1.0e-8)
    keys(reference) == keys(candidate) ||
        error("Julia and Reactant returned different output fields")

    max_absolute_difference = 0.0
    max_relative_difference = 0.0
    worst_output_field = "none"

    for name in keys(reference)
        expected = getproperty(reference, name)
        actual = getproperty(candidate, name)
        if expected === nothing
            actual === nothing || error("$name differs between Julia and Reactant")
            continue
        end

        size(expected) == size(actual) || error("$name has mismatched dimensions")
        if eltype(expected) <: Bool
            expected == actual || error("$name differs between Julia and Reactant")
            continue
        end

        all(isfinite, actual) || error("$name contains non-finite Reactant values")
        isapprox(expected, actual; rtol, atol) ||
            error("$name differs between Julia and Reactant")

        for index in eachindex(expected, actual)
            expected_value = Float64(expected[index])
            actual_value = Float64(actual[index])
            absolute_difference = abs(expected_value - actual_value)
            relative_difference =
                absolute_difference / max(abs(expected_value), atol)
            max_absolute_difference =
                max(max_absolute_difference, absolute_difference)
            if relative_difference > max_relative_difference
                max_relative_difference = relative_difference
                worst_output_field = string(name)
            end
        end
    end

    @info "Correctness check passed" rtol atol max_absolute_difference max_relative_difference worst_output_field
    return (;
        passed=true,
        max_absolute_difference,
        max_relative_difference,
        worst_output_field,
        rtol,
        atol,
    )
end

function timed_input_placement(cpu_args)
    start_time = time_ns()
    ra_args = Reactant.to_rarray(cpu_args)
    Reactant.synchronize(ra_args)
    elapsed_seconds = (time_ns() - start_time) / 1.0e9
    return ra_args, elapsed_seconds
end

function measure_julia_overheads(fn, cpu_args)
    GC.gc(true)
    first_call = @timed fn(cpu_args...)
    force_host_completion(first_call.value)
    first_execution_seconds = max(first_call.time - first_call.compile_time, 0.0)
    overheads = (;
        input_placement_seconds=0.0,
        compile_seconds=first_call.compile_time,
        first_execution_seconds,
        materialization_seconds=0.0,
        one_shot_seconds=first_call.time,
    )
    return first_call.value, overheads
end

function measure_reactant_overheads(fn, cpu_args)
    ra_args, input_placement_seconds = timed_input_placement(cpu_args)
    compile_options = Reactant.__compile_options_with_updated_sync(
        Reactant.CompileOptions(),
        true,
    )

    start_time = time_ns()
    compiled_fn = Reactant.compile(fn, ra_args; compile_options)
    compile_seconds = (time_ns() - start_time) / 1.0e9

    start_time = time_ns()
    first_output = compiled_fn(ra_args...)
    Reactant.synchronize(first_output)
    first_execution_seconds = (time_ns() - start_time) / 1.0e9

    start_time = time_ns()
    host_output = materialize_output(first_output)
    force_host_completion(host_output)
    materialization_seconds = (time_ns() - start_time) / 1.0e9

    one_shot_seconds =
        input_placement_seconds + compile_seconds + first_execution_seconds +
        materialization_seconds
    overheads = (;
        input_placement_seconds,
        compile_seconds,
        first_execution_seconds,
        materialization_seconds,
        one_shot_seconds,
    )
    return ra_args, host_output, overheads
end

function run_bayesmmfwd_benchmark!(
    results::Dict,
    backend::String;
    dataset="data/SIDES_Bethermin2017_short2.csv",
    param_path="data/SIDES_from_original.par",
    nrows=nothing,
    filters=false,
)
    params = BayesMMfwd.load_par_file(param_path)
    catalog = BayesMMfwd.load_sides_csv(dataset, nrows)
    inputs = BayesMMfwd.build_forward_inputs(catalog)
    parameter_data = BayesMMfwd.build_forward_parameters(params; filters)
    noise = BayesMMfwd.make_simulation_noise(length(inputs.redshift))

    cpu_args = (inputs, parameter_data.numeric, noise)

    n_gal = length(inputs.redshift)
    mode = filters ? "forward_filters" : "forward"
    benchmark_name = "BayesMMfwd [$n_gal galaxies]/$mode"

    julia_overheads = nothing
    julia_first_output = nothing
    if n_gal == 1_000_000 && backend == "CPU"
        julia_first_output, julia_overheads =
            measure_julia_overheads(BayesMMfwd.forward_model, cpu_args)
    end

    overheads = nothing
    overhead_host_output = nothing
    if n_gal == 1_000_000
        ra_args, overhead_host_output, overheads =
            measure_reactant_overheads(BayesMMfwd.forward_model, cpu_args)
    else
        ra_args = Reactant.to_rarray(cpu_args)
        Reactant.synchronize(ra_args)
    end

    reactant_output = run_benchmark!(
        results,
        backend,
        benchmark_name,
        BayesMMfwd.forward_model,
        cpu_args,
        ra_args;
        configs=[BenchmarkConfiguration("Default")],
    )
    reference = isnothing(julia_first_output) ?
                BayesMMfwd.forward_model(cpu_args...) : julia_first_output
    candidate = if isnothing(overhead_host_output)
        materialized = materialize_output(reactant_output)
        force_host_completion(materialized)
        materialized
    else
        overhead_host_output
    end
    correctness = check_forward_output(reference, candidate)

    if !isnothing(overheads)
        reactant_name = string(benchmark_name, "/", backend, "/Default")
        resident_execution_seconds = results["Runtime (s)"][reactant_name]
        overheads = merge(overheads, (; resident_execution_seconds))
    end
    if !isnothing(julia_overheads)
        julia_name = string(benchmark_name, "/CPU/Julia")
        resident_execution_seconds = results["Runtime (s)"][julia_name]
        julia_overheads = merge(julia_overheads, (; resident_execution_seconds))
    end
    return (; n_gal, benchmark_name, overheads, julia_overheads, correctness)
end
