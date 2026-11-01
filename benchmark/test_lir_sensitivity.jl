using Test

include("benchmark_lir_sensitivity.jl")

@testset "Per-galaxy LIR sensitivity" begin
    raw = BayesMMfwd.load_par_file("data/SIDES_from_original.par")
    params = BayesMMfwd.build_forward_parameters(raw; filters=false).numeric
    n = 8
    redshift = fill(1.0, n)
    mstar = fill(1.0e10, n)
    inputs = BayesMMfwd.build_forward_inputs((; redshift, Mstar=mstar))
    noise = merge(BayesMMfwd.make_simulation_noise(n; seed=LIR_SEED), (
        q_uniform=zeros(n),
        sb_uniform=ones(n),
        sfr_uniform=[0.01, 0.1, 0.25, 0.4, 0.6, 0.75, 0.9, 0.99],
    ))
    args = ([params.sfr.sigma_MS], redshift, mstar, params.sfr, noise, raw["SFR2LIR"])
    luminosity = lir_per_galaxy(args...)
    derivative = lir_sigma_sensitivity(args...)

    # Verify that selecting the SFR-to-LIR dependency preserves the public
    # forward model's actual LIR output at more than one parameter value.
    for sigma in (0.2, 0.3, 0.4)
        varied = merge(params, (sfr=merge(params.sfr, (sigma_MS=sigma,)),))
        full_output = BayesMMfwd.forward_model(inputs, varied, noise)
        @test lir_per_galaxy([sigma], args[2:end]...) ≈ full_output.LIR
    end
    @test length(derivative) == n
    @test all(isfinite, derivative)
    @test any(<(0), derivative)
    @test any(>(0), derivative)
    for step in (1.0e-4, 5.0e-5)
        reference = lir_finite_difference(args; step)
        check = lir_comparison(reference, derivative, luminosity; rtol=2.0e-5, scaled_atol=1.0e-7)
        @test check["passed"]
    end
    passive_noise = merge(noise, (q_uniform=ones(n),))
    passive_args = (args[1:4]..., passive_noise, args[6])
    @test all(iszero, lir_per_galaxy(passive_args...))
    @test all(iszero, lir_sigma_sensitivity(passive_args...))

    # A malformed vector or deliberately corrupted element must invalidate
    # the point, even when all other galaxy derivatives agree.
    corrupted = copy(derivative)
    corrupted[4] += 0.01luminosity[4]
    @test !lir_comparison(derivative, corrupted, luminosity; rtol=2.0e-6, scaled_atol=1.0e-8)["passed"]
    @test_throws ErrorException lir_comparison(derivative, derivative[1:2], luminosity; rtol=2.0e-6, scaled_atol=1.0e-8)
    @test_throws ErrorException lir_finite_difference(args; step=0.2)
end
