# ============================================================
# 1. IMPORTS AND SUPPORTING FILES
# ============================================================

using Random
using DelimitedFiles
using Statistics
using Reactant
using SpecialFunctions: erf
using Reactant: ProbProg, ReactantRNG, ConcreteRNumber, ConcreteRArray

include(joinpath(@__DIR__, "..", "src", "io.jl"))
include(joinpath(@__DIR__, "..", "src", "reactant_pipeline.jl"))
include(joinpath(@__DIR__, "..", "src", "pipeline_host.jl"))


# ============================================================
# 2. CONFIGURATION
# ============================================================

# false = sample m0, a0, and log_sigma
# true  = fix a0 = 1.5 and sample m0 and log_sigma
const TWO_ADDR = false

const N_GALAXIES = 500
const NUM_WARMUP = 1000
const NUM_SAMPLES = 1000


# ============================================================
# 3. SYNTHETIC FIXTURE
# ============================================================

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
        for u_index in eachindex(umean_grid),
            z_index in 1:n_z_filter
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
        rJup1_Birkin=Float64[
            0.9, 0.6, 0.32, 0.35, 0.3, 0.22, 0.22 * (7 / 8)^2
        ],
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

function generate_forward_model_data(; obs_sigma=0.05)

    fixture = synthetic_fixture()
    base_params = fixture.params

    # Generate N_GALAXIES synthetic inputs.
    n = N_GALAXIES
    drng = MersenneTwister(BENCHMARK_SEED + 1)

    redshift = 3.0 .* rand(drng, n)
    mstar = 10.0 .^ (9.0 .+ 3.0 .* rand(drng, n))

    inputs = (
        redshift,
        Mstar=mstar,
        dlum_mpc=100.0 .* (1.0 .+ redshift),
        dlum_m=100.0 .* (1.0 .+ redshift) .* MPC_TO_M,
    )

    # Generate noise arrays with matching dimensions.
    noise = make_simulation_noise(n)

    # Ground-truth parameters.
    true_m0 = 0.5
    true_a0 = 1.5
    true_sigma = 0.3

    true_sfr_params = merge(
        base_params.sfr,
        (;
            m0=true_m0,
            a0=true_a0,
            sigma_MS=true_sigma,
        )
    )

    true_params = merge(
        base_params,
        (; sfr=true_sfr_params)
    )

    # Generate observations using the complete forward model.
    outputs = forward_model(inputs, true_params, noise)

    # Keep only galaxies with positive SFR.
    sf_indices = findall(outputs.SFR .> 0)

    y_true = log10.(outputs.SFR[sf_indices])

    rng = MersenneTwister(1234)
    y_obs = y_true .+ obs_sigma .* randn(rng, length(y_true))

    println("Star-forming galaxies: ", length(sf_indices))
    println("Zero-SFR galaxies excluded: ", length(outputs.SFR) - length(sf_indices))
    println("Generated observations for ", length(y_obs), " galaxies.")

    return (;
        inputs,
        base_params,
        noise,
        y_obs,
        sf_indices,
        obs_sigma,
        true_m0,
        true_a0,
        true_sigma,
    )
end


# ============================================================
# 4. FULL FORWARD-MODEL PROBABILISTIC MODEL
# ============================================================

# function impulse_model(rng, inputs, base_params, noise)

#     # Sample the three parameters.
#     _, m0 = ProbProg.sample(
#         rng, ProbProg.Normal(0.0, 2.0, (1,)); symbol=:m0
#     )

#     _, a0 = ProbProg.sample(
#         rng, ProbProg.Normal(0.0, 2.0, (1,)); symbol=:a0
#     )

#     _, sigma_MS = ProbProg.sample(
#         rng, ProbProg.Normal(0.0, 1.0, (1,)); symbol=:sigma_MS
#     )

#     # Replace the sampled parameters in the SFR parameter set.
#     sfr_params = merge(
#         base_params.sfr,
#         (; m0, a0, sigma_MS)
#     )

#     params = merge(base_params, (; sfr=sfr_params))

#     # Run the complete forward model.
#     outputs = forward_model(inputs, params, noise)

#     return outputs
# end


# ============================================================
# 5. PRIOR-PREDICTIVE SIMULATION
# ============================================================

function run_prior_predictive()

    println("\n========== PRIOR-PREDICTIVE SIMULATION ==========")

    fixture = synthetic_fixture()
    inputs, params, noise = fixture.inputs, fixture.params, fixture.noise

    inputs_r = Reactant.to_rarray(inputs)
    params_r = Reactant.to_rarray(params)
    noise_r = Reactant.to_rarray(noise)

    seed = Reactant.to_rarray(UInt64[1, 4])
    rng = ReactantRNG(seed)

    result = @jit optimize=:probprog ProbProg.simulate(
        rng, forward_model_impulse, inputs_r, params_r, noise_r
    )

    trace, weight, (rng_out, outputs) = result

    println("Sampled (m0, a0, sigma_MS): ", Array(trace))
    println("Log-weight: ", Array(weight)[])
    println("SFR:   ", Array(outputs.SFR))
    println("issb:  ", Array(outputs.issb))
    println("ICO10: ", Array(outputs.ICO10))

    return outputs
end


# ============================================================
# 6. GENERATE SYNTHETIC OBSERVATIONS FOR NUTS
# ============================================================

function generate_synthetic_observations()

    println("\n========== GENERATING SYNTHETIC OBSERVATIONS ==========")

    n = N_GALAXIES

    drng = MersenneTwister(BENCHMARK_SEED + 1)

    z = 3.0 .* rand(drng, n)
    mstar = 10.0 .^ (9.0 .+ 3.0 .* rand(drng, n))

    nz = make_simulation_noise(n)
    p = synthetic_fixture().params.sfr

    sfr_out = generate_sfr(z, mstar, p, nz)

    # Keep only star-forming galaxies.
    sf = findall(.!sfr_out.qflag)

    zs = z[sf]
    ms = mstar[sf]
    issb = Float64.(sfr_out.issb[sf])

    # Calculate the predictor variables.
    m = log10.(ms .* p.Chab2Salp_num ./ 1.0e9)
    r = log10.(1.0 .+ zs)

    expr = max.(m .- p.m1 .- p.a2 .* r, 0.0)

    corr = p.corr_zmean_lowzcorr .*
           (p.zmax_lowzcorr .- min.(zs, p.zmax_lowzcorr)) ./
           (p.zmax_lowzcorr - p.zmean_lowzcorr)

    offset = m .-
             p.a1 .* expr .^ 2 .-
             log10(p.Chab2Salp_num) .+
             corr .+
             p.logx0 .+
             issb .* (p.logBsb - p.logx0)

    # Ground-truth parameters.
    true_m0 = 0.5
    true_a0 = 1.5
    true_sigma = 0.3

    # Generate observations from the simplified regression model.
    y_obs = offset .-
            true_m0 .+
            true_a0 .* r .+
            true_sigma .* randn(drng, length(r))

    # Alternative: use the forward-model SFR outputs directly.
    # y_obs = log10.(sfr_out.SFR[sf])

    println("Star-forming galaxies: ", length(y_obs))

    return (; r, offset, y_obs, true_m0, true_a0, true_sigma)
end

# ============================================================
# 7. FULL FORWARD-MODEL PROBABILISTIC MODEL
# ============================================================

function forward_model_probprog(
    rng,
    inputs,
    base_params,
    noise,
    obs_sigma,
    sf_indices,
)
    # Sample parameters.
    _, m0 = ProbProg.sample(
        rng, ProbProg.Normal(0.0, 2.0, (1,)); symbol=:m0
    )

    _, a0 = ProbProg.sample(
        rng, ProbProg.Normal(0.0, 2.0, (1,)); symbol=:a0
    )

    _, log_sigma = ProbProg.sample(
        rng, ProbProg.Normal(-1.0, 1.0, (1,)); symbol=:log_sigma
    )

    sigma_MS = exp.(log_sigma)

    # Update the SFR parameters.
    sfr_params = merge(
        base_params.sfr,
        (; m0, a0, sigma_MS)
    )

    params = merge(base_params, (; sfr=sfr_params))

    # Run the complete forward model on all 8 galaxies.
    outputs = forward_model(inputs, params, noise)

    # Keep only the star-forming galaxies.
    predicted_sfr = outputs.SFR[sf_indices]
    predicted_log_sfr = log10.(predicted_sfr)

    # Observation likelihood has length 6, matching y_obs.
    n = length(predicted_log_sfr)

    _, y = ProbProg.sample(
        rng,
        ProbProg.Normal(
            predicted_log_sfr,
            obs_sigma .* ones(n),
            (n,)
        );
        symbol=:y
    )

    return y
end


# ============================================================
# 8. NUTS PROGRAM
# ============================================================

function nuts_program(
    rng,
    model,
    inputs,
    base_params,
    noise,
    obs_sigma,
    sf_indices,
    step_size,
    inverse_mass_matrix,
    num_warmup::Int,
    num_samples::Int,
    obs_tensor,
    constrained_addresses,
    selection,
)

    trace, _, _ = ProbProg.generate(
        rng,
        obs_tensor,
        model,
        inputs,
        base_params,
        noise,
        obs_sigma,
        sf_indices;
        constrained_addresses
    )

    trace, diagnostics, log_densities, _, state = ProbProg.mcmc(
    rng,
    trace,
    model,
    inputs,
    base_params,
    noise,
    obs_sigma,
    sf_indices;
    selection,
    algorithm=:NUTS,
    inverse_mass_matrix,
    step_size,
    num_warmup,
    num_samples,
    adapt_step_size=true,
    adapt_mass_matrix=true,
)

return trace, diagnostics, log_densities, state
end


# ============================================================
# 9. RUN NUTS INFERENCE
# ============================================================

function run_forward_model_nuts()

    println("\n========== FULL FORWARD-MODEL NUTS ==========")

    # Generate observations from the reference parameter values.
    data = generate_forward_model_data(obs_sigma=0.05)

    # Convert model inputs to Reactant arrays.
    inputs_r = Reactant.to_rarray(data.inputs)
    params_r = Reactant.to_rarray(data.base_params)
    noise_r = Reactant.to_rarray(data.noise)
    sf_indices_r = Reactant.to_rarray(data.sf_indices)

    # Constrain the observation site and initialize the parameters.
    obs = ProbProg.Constraint(
        :y => data.y_obs,
    )

    selection = ProbProg.select(
        ProbProg.Address(:m0),
        ProbProg.Address(:a0),
        ProbProg.Address(:log_sigma),
    )

    constrained_addresses = ProbProg.extract_addresses(obs)
    obs_tensor = ProbProg.flatten_constraint(obs)

    # Identity inverse mass matrix.
    k = 3
    inverse_mass_matrix = ConcreteRArray([
        i == j ? 1.0 : 0.0
        for i in 1:k, j in 1:k
    ])

    rng = ReactantRNG(
        Reactant.to_rarray(UInt64[1234, 5678])
    )

    # Run NUTS.
    trace, diagnostics, log_densities, state = @jit optimize=:probprog nuts_program(
        rng,
        forward_model_probprog,
        inputs_r,
        params_r,
        noise_r,
        ConcreteRNumber(data.obs_sigma),
        sf_indices_r,
        ConcreteRNumber(0.05),
        inverse_mass_matrix,
        NUM_WARMUP,
        NUM_SAMPLES,
        obs_tensor,
        constrained_addresses,
        selection,
    )

    D = Array(diagnostics)
    L = Array(log_densities)

    println("Divergent transitions: ", count(D[:, 2]), " / ", size(D, 1))
    println("Log density range: ", extrema(L))
    println("Final sampler position: ", Array(state.position))
    println("Final step size: ", Array(state.step_size))
    println("Final potential energy: ", Array(state.potential_energy))
 
    S = Array(trace)
    println("Trace size: ", size(S))
    println("First 10 samples:")
    display(S[1:min(10, end), :])

    println("Parameter ranges:")
    println("m0: ", extrema(S[:, 1]))
    println("a0: ", extrema(S[:, 2]))
    println("log_sigma: ", extrema(S[:, 3]))

    # Convert log_sigma samples back to sigma_MS.
    post = hcat(
        S[:, 1],
        S[:, 2],
        exp.(S[:, 3]),
    )

    # Save samples.
    output_path = joinpath(@__DIR__, "forward_model_nuts_samples.csv")
    writedlm(output_path, post, ',')

    println("Saved samples to: ", output_path)

    # Print posterior summaries.
    for (name, col, truth) in zip(
        ["m0", "a0", "sigma_MS"],
        eachcol(post),
        [data.true_m0, data.true_a0, data.true_sigma],
    )
        println(
            rpad(name, 10),
            "mean=", round(mean(col), digits=3),
            "  std=", round(std(col), digits=3),
            "  true=", truth,
        )
    end

    D = Array(diagnostics)
    println("Diagnostics column means: ", mean(D, dims=1))

    return post, diagnostics
end

# ============================================================
# 10. POSTERIOR SUMMARIES AND OUTPUT
# ============================================================

function summarize_results(result)

    println("\n========== POSTERIOR RESULTS ==========")

    S = Array(result.trace)

    if TWO_ADDR
        # Columns: m0, log_sigma
        post = hcat(
            S[:, 1],
            exp.(S[:, 2])
        )

        parameter_names = ["m0", "sigma_MS"]
        truths = [result.true_m0, result.true_sigma]

    else
        # Columns: m0, a0, log_sigma
        post = hcat(
            S[:, 1],
            S[:, 2],
            exp.(S[:, 3])
        )

        parameter_names = ["m0", "a0", "sigma_MS"]
        truths = [result.true_m0, result.true_a0, result.true_sigma]
    end

    # Save posterior samples.
    output_path = joinpath(@__DIR__, "nuts_samples.csv")
    writedlm(output_path, post, ',')

    println("Saved posterior samples to: ", output_path)

    # Print posterior statistics.
    for (name, col, truth) in zip(
        parameter_names,
        eachcol(post),
        truths
    )
        println(
            rpad(name, 10),
            "mean=", round(mean(col), digits=3),
            "  std=", round(std(col), digits=3),
            "  true=", truth
        )
    end

    # Print diagnostics.
    D = Array(result.diagnostics)
    println("Diagnostics column means: ", mean(D, dims=1))

    return post
end


# ============================================================
# 11. MAIN
# ============================================================


function main()

    # # Run the full forward model with sampled parameters.
    run_prior_predictive()

    # # Run parameter inference with NUTS.
    # nuts_result = run_nuts()

    # # Summarize posterior samples.
    # summarize_results(nuts_result)

    # println("\n========== DONE ==========")
    #run_forward_model_nuts()

end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

