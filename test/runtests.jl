using Test
include("maps.jl")
include("cubes.jl")
import Reactant
using DataFrames: DataFrame, names, nrow
using HDF5: h5open
using SkyMatch:
    C_M_PER_S,
    MPC_TO_M,
    add_output_columns!,
    forward_model,
    gen_outputs,
    interpolate_sed_fluxes,
    linear_interp_sorted,
    load_filter_table,
    load_par_file,
    make_simulation_noise,
    normal_quantile,
    searchsortedlast_vector,
    sed_uindex

Reactant.set_default_backend("cpu")

function materialize_test_output(value::NamedTuple)
    names = keys(value)
    values = map(materialize_test_output, Tuple(value))
    return NamedTuple{names}(values)
end

materialize_test_output(value::Tuple) = map(materialize_test_output, value)
materialize_test_output(value::AbstractArray) = Array(value)
materialize_test_output(value) = value

function collect_output_arrays!(result, prefix, value::NamedTuple)
    for name in keys(value)
        child_prefix = isempty(prefix) ? String(name) : "$prefix.$name"
        collect_output_arrays!(result, child_prefix, getproperty(value, name))
    end
    return result
end

function collect_output_arrays!(result, prefix, value::AbstractArray)
    push!(result, prefix => value)
    return result
end

collect_output_arrays!(result, prefix, value) = result

function synthetic_fixture()
    redshift = Float64[0.01, 0.1, 0.3, 0.7, 1.5, 3.0, 8.0, 20.0]
    n_gal = length(redshift)
    inputs = (
        redshift,
        Mstar=10.0 .^ range(9.0, 12.0; length=n_gal),
        dlum_mpc=100.0 .* (1.0 .+ redshift),
        dlum_m=100.0 .* (1.0 .+ redshift) .* MPC_TO_M,
    )

    sfr = (
        Chab2Salp_num=10.0^0.24,
        Mt0=10.5296,
        alpha1=0.223213,
        alpha2=0.0912848,
        sigma0=0.848799,
        beta1=0.0417672,
        beta2=-0.0158969,
        qfrac0=0.101681,
        gamma=-1.03861,
        m1=0.36,
        a2=2.5,
        m0=0.5,
        a0=1.5,
        a1=0.3,
        corr_zmean_lowzcorr=-0.1,
        zmax_lowzcorr=0.5,
        zmean_lowzcorr=0.2185,
        Psb_hz=0.03,
        slope_Psb=-0.015,
        z_Psb_knee=1.0,
        sigma_MS=0.3,
        logx0=log10(0.87),
        logBsb=log10(5.3),
        SFR_max=1000.0,
    )

    z_grid = Float64[0.0, 0.5, 2.0, 10.0]
    mu_grid = Float64[1.0, 1.5, 2.0, 3.0]
    base_cdf = Float64[0.0, 0.4, 0.8, 1.0]
    magnification = (
        z_grid,
        mu_grid,
        Psupmu=hcat(
            base_cdf,
            base_cdf .^ 0.9,
            base_cdf .^ 0.8,
            base_cdf .^ 0.7,
        ),
    )

    umean_grid = collect(0.1:0.1:40.0)
    sed_lambda = Float64[1.0, 10.0, 100.0, 1000.0]
    sed_ms = [
        (0.1 + 0.01 * u_index) * (1.0 + 0.2 * lambda_index)
        for lambda_index in eachindex(sed_lambda),
        u_index in eachindex(umean_grid)
    ]
    sed_sb = 1.5 .* sed_ms
    ratio_ms = collect(0.8 .- 0.0002 .* eachindex(umean_grid))
    ratio_sb = collect(0.9 .- 0.0001 .* eachindex(umean_grid))
    dust = (
        UmeanSB=31.0,
        UmeanMSz0=5.0,
        alphaMS=0.25,
        zlimMS=4.0,
        sigma_logUmean=0.2,
        SFR2LIR=1.0e10,
        lambda_list=Float64[24.0, 250.0],
        sed_tables=(
            lambda=sed_lambda,
            dU=0.1,
            Umean=umean_grid,
            nuLnu_MS=sed_ms,
            nuLnu_SB=sed_sb,
        ),
        ratio_tables=(
            Umean=umean_grid,
            dU=0.1,
            LFIR_LIR_ratio_MS=ratio_ms,
            LFIR_LIR_ratio_SB=ratio_sb,
        ),
    )

    n_z_filter = 20
    filter_ms = [
        0.01 * u_index + 0.001 * z_index
        for u_index in eachindex(umean_grid), z_index in 1:n_z_filter
    ]
    filter_table = (
        dU=0.1,
        dlog1plusz=0.05,
        Nlog1plusz=n_z_filter,
        Umean=umean_grid,
        Snu_LIR_MS=filter_ms,
        Snu_LIR_SB=2.0 .* filter_ms,
    )

    idiffuse = Float64[1.0, 2.25, 2.30, 2.05, 1.40, 1.10, 0.75, 0.45]
    iclump = Float64[1.0, 2.87, 5.25, 10.87, 16.12, 9.37, 5.25, 2.62]
    lines = (
        sigma_dex_CO10=0.2,
        nu_CO=115.271208,
        Idiffuse=idiffuse,
        Iclump=iclump,
        SLED_SB_Birkin=true,
        rJup1_Birkin=Float64[0.9, 0.6, 0.32, 0.35, 0.3, 0.22, 0.22*(7/8)^2],
        nu_CII=1900.54,
        sigma_dex_CII=0.2,
        a_CI10=1.07,
        b_CI10=0.14,
        sigma_CI10=0.2,
        a_CI21=0.63,
        b_CI21=0.17,
        sigma_CI21=0.19,
        nu_CI10=492.16,
        nu_CI21=809.34,
    )
    params = (; sfr, magnification, dust, filters=(filter_table,), lines)
    noise = make_simulation_noise(n_gal)
    return (; inputs, params, noise)
end

@testset "Reactant-compatible forward model" begin
    fixture = synthetic_fixture()

    @test normal_quantile(0.5) ≈ 0.0 atol = 1.0e-12
    @test normal_quantile(0.975) ≈ 1.959963986 atol = 1.0e-8
    sorted_values = Float64[0.0, 0.5, 0.5, 2.0, 10.0]
    queries = Float64[-1.0, 0.0, 0.5, 1.0, 10.0, 11.0]
    @test searchsortedlast_vector(sorted_values, queries) ==
          searchsortedlast.(Ref(sorted_values), queries)
    @test sed_uindex(Float64[0.1, 5.0, 31.0], fixture.params.dust.sed_tables) ==
          Int64[1, 50, 310]

    julia_output = forward_model(fixture.inputs, fixture.params, fixture.noise)
    @test size(julia_output.monochromatic_fluxes) == (8, 2)
    @test size(julia_output.filter_fluxes) == (8, 1)
    @test size(julia_output.co_fluxes) == (8, 7)
    @test all(julia_output.SFR .<= fixture.params.sfr.SFR_max)
    @test all(julia_output.SFR[julia_output.qflag] .== 0.0)
    @test all(.!julia_output.issb[julia_output.qflag])

    sed_shape = interpolate_sed_fluxes(
        fixture.params.dust.lambda_list,
        fixture.params.dust.sed_tables,
        fixture.inputs.redshift,
        julia_output.Umean,
        julia_output.issb,
    )
    expected_sed_shape = [
        begin
            u_index = sed_uindex(
                [julia_output.Umean[galaxy]],
                fixture.params.dust.sed_tables,
            )[1]
            sed = julia_output.issb[galaxy] ?
                  fixture.params.dust.sed_tables.nuLnu_SB[:, u_index] :
                  fixture.params.dust.sed_tables.nuLnu_MS[:, u_index]
            lambda_rest = wavelength / (1.0 + fixture.inputs.redshift[galaxy])
            nu_rest = C_M_PER_S / (lambda_rest * 1.0e-6)
            linear_interp_sorted(
                lambda_rest,
                fixture.params.dust.sed_tables.lambda,
                sed,
            ) / nu_rest
        end
        for galaxy in eachindex(fixture.inputs.redshift),
        wavelength in fixture.params.dust.lambda_list
    ]
    @test sed_shape ≈ expected_sed_shape

    arrays = collect_output_arrays!(Pair{String,Any}[], "", julia_output)
    @test all(
        !(eltype(array) <: AbstractFloat) || all(isfinite, array)
        for (_, array) in arrays
    )

    inputs_reactant = Reactant.to_rarray(fixture.inputs)
    params_reactant = Reactant.to_rarray(fixture.params)
    noise_reactant = Reactant.to_rarray(fixture.noise)
    compiled_forward = Reactant.@compile sync=true forward_model(
        inputs_reactant,
        params_reactant,
        noise_reactant,
    )
    reactant_output = compiled_forward(
        inputs_reactant,
        params_reactant,
        noise_reactant,
    )
    Reactant.synchronize(reactant_output)
    reactant_host_output = materialize_test_output(reactant_output)
    reference_arrays = collect_output_arrays!(Pair{String,Any}[], "", julia_output)
    reactant_arrays = Dict(collect_output_arrays!(
        Pair{String,Any}[],
        "",
        reactant_host_output,
    ))
    @test Set(first.(reference_arrays)) == Set(keys(reactant_arrays))
    for (name, expected) in reference_arrays
        actual = reactant_arrays[name]
        @test size(actual) == size(expected)
        if eltype(expected) <: Bool
            @test actual == expected
        else
            @test isapprox(actual, expected; rtol=2.0e-6, atol=1.0e-8)
        end
    end
end

@testset "catalog and host output" begin
    fixture = synthetic_fixture()
    output = forward_model(fixture.inputs, fixture.params, fixture.noise)
    catalog = DataFrame(
        redshift=fixture.inputs.redshift,
        ra=zeros(8),
        dec=zeros(8),
        Mhalo=ones(8),
        Mstar=fixture.inputs.Mstar,
    )
    add_output_columns!(
        catalog,
        output,
        fixture.params.dust.lambda_list,
        ["SYNTHETIC"],
    )
    @test "SSYNTHETIC" in names(catalog)
    @test "ICO87" in names(catalog)
    @test nrow(catalog) == 8

    source_params = load_par_file(joinpath(
        @__DIR__,
        "..",
        "data",
        "SIDES_from_original.par",
    ))
    mktempdir() do directory
        filter_path = joinpath(directory, "SYNTHETIC.h5")
        filter_table = fixture.params.filters[1]
        h5open(filter_path, "w") do file
            file["dU"] = Float32(filter_table.dU)
            file["dlog1plusz"] = Float32(filter_table.dlog1plusz)
            file["Nlog1plusz"] = Int32(filter_table.Nlog1plusz)
            file["Umean"] = Float32.(filter_table.Umean)
            file["Snu_LIR_MS"] = Float32.(permutedims(filter_table.Snu_LIR_MS))
            file["Snu_LIR_SB"] = Float32.(permutedims(filter_table.Snu_LIR_SB))
        end
        loaded_filter = load_filter_table(filter_path)
        @test loaded_filter.Umean isa Vector{Float64}
        @test isapprox(
            loaded_filter.Snu_LIR_MS,
            filter_table.Snu_LIR_MS;
            rtol=1.0e-6,
        )
        @test isapprox(
            loaded_filter.Snu_LIR_SB,
            filter_table.Snu_LIR_SB;
            rtol=1.0e-6,
        )

        source_params["output_path"] = directory
        source_params["run_name"] = "reactant_smoke"
        source_params["gen_fits"] = true
        @test gen_outputs(catalog, source_params)
        @test isfile(joinpath(directory, "reactant_smoke.fits"))
    end
end
