using Reactant
using Reactant: ProbProg
using SpecialFunctions: erf

const MPC_TO_M = 3.0856775814913673e22
const L_SUN_W = 3.828e26
const C_M_PER_S = 299792458.0

"""
    normal_quantile(p)

Acklam's rational approximation of the standard-normal quantile.  Keeping this
small kernel in Base arithmetic makes upper-truncated SFR sampling available to
both ordinary Julia arrays and Reactant arrays (Reactant does not currently
provide `erfinv` for traced numbers).
"""
function normal_quantile(p)
    p_safe = clamp(p, 1.0e-15, 1.0 - 1.0e-15)
    p_low = 0.02425
    p_high = 1.0 - p_low

    a1 = -3.969683028665376e1
    a2 = 2.209460984245205e2
    a3 = -2.759285104469687e2
    a4 = 1.383577518672690e2
    a5 = -3.066479806614716e1
    a6 = 2.506628277459239

    b1 = -5.447609879822406e1
    b2 = 1.615858368580409e2
    b3 = -1.556989798598866e2
    b4 = 6.680131188771972e1
    b5 = -1.328068155288572e1

    c1 = -7.784894002430293e-3
    c2 = -3.223964580411365e-1
    c3 = -2.400758277161838
    c4 = -2.549732539343734
    c5 = 4.374664141464968
    c6 = 2.938163982698783

    d1 = 7.784695709041462e-3
    d2 = 3.224671290700398e-1
    d3 = 2.445134137142996
    d4 = 3.754408661907416

    q_tail = sqrt(-2.0 * log(min(p_safe, 1.0 - p_safe)))
    tail = (((((c1 * q_tail + c2) * q_tail + c3) * q_tail + c4) *
             q_tail + c5) * q_tail + c6) /
           ((((d1 * q_tail + d2) * q_tail + d3) * q_tail + d4) * q_tail + 1.0)
    tail = ifelse(p_safe < p_low, tail, -tail)

    q_center = p_safe - 0.5
    r_center = q_center * q_center
    center = (((((a1 * r_center + a2) * r_center + a3) * r_center + a4) *
               r_center + a5) * r_center + a6) * q_center /
             (((((b1 * r_center + b2) * r_center + b3) * r_center + b4) *
               r_center + b5) * r_center + 1.0)

    return ifelse((p_safe < p_low) | (p_safe > p_high), tail, center)
end

normal_cdf(x) = 0.5 * (1.0 + erf(x / sqrt(2.0)))

"""
    generate_sfr(redshift, mstar, p, noise)

Generate quenched flags, SFRs, and starburst flags.  The normal draw is sampled
directly from the upper-truncated distribution, which is distributionally
equivalent to the rejection loop in pySIDES and has static control flow.
"""
function generate_sfr(redshift, mstar, p, noise)
    (; Chab2Salp_num, Mt0, alpha1, alpha2, sigma0, beta1, beta2, qfrac0, gamma,
        m1, a2, m0, a0, a1, corr_zmean_lowzcorr, zmax_lowzcorr, zmean_lowzcorr,
        Psb_hz, slope_Psb, z_Psb_knee, sigma_MS, logx0, logBsb, SFR_max) = p

    Mtz = Mt0 .+ alpha1 .* redshift .+ alpha2 .* redshift .^ 2
    sigmaz = sigma0 .+ beta1 .* redshift .+ beta2 .* redshift .^ 2
    qfrac0z = qfrac0 .* (1.0 .+ redshift) .^ gamma
    prob_sf = (1.0 .- qfrac0z) .* 0.5 .*
              (1.0 .- erf.((log10.(mstar) .- Mtz) ./ sigmaz))
    qflag = noise.q_uniform .> prob_sf
    mask_sf = .!qflag

    m = log10.(mstar .* Chab2Salp_num ./ 1.0e9)
    r = log10.(1.0 .+ redshift)
    expr = max.(m .- m1 .- a2 .* r, 0.0)
    log_sfr_ms = m .- m0 .+ a0 .* r .- a1 .* expr .^ 2 .-
                 log10(Chab2Salp_num)
    log_sfr_ms = log_sfr_ms .+
        corr_zmean_lowzcorr .* (zmax_lowzcorr .- min.(redshift, zmax_lowzcorr)) ./
        (zmax_lowzcorr - zmean_lowzcorr)

    psb = Psb_hz .+ slope_Psb .* (z_Psb_knee .- min.(redshift, z_Psb_knee))
    issb_draw = noise.sb_uniform .< psb
    log_sfr_mean = log_sfr_ms .+ logx0 .+
                   issb_draw .* (logBsb - logx0)

    upper_sigma = (log10(SFR_max) .- log_sfr_mean) ./ sigma_MS
    truncated_probability = noise.sfr_uniform .* normal_cdf.(upper_sigma)
    truncated_normal = normal_quantile.(truncated_probability)
    sfr_draw = 10.0 .^ (log_sfr_mean .+ sigma_MS .* truncated_normal)

    sfr = ifelse.(mask_sf, min.(sfr_draw, SFR_max), 0.0)
    issb = ifelse.(mask_sf, issb_draw, false)
    return (; qflag, SFR=sfr, issb)
end

function linear_interp_sorted(x, xp, fp)
    n = length(xp)
    i = searchsortedlast(xp, x)
    i_lo = ifelse(i <= 0, 1, ifelse(i >= n, n - 1, i))
    i_hi = i_lo + 1

    x1 = @allowscalar xp[i_lo]
    x2 = @allowscalar xp[i_hi]
    y1 = @allowscalar fp[i_lo]
    y2 = @allowscalar fp[i_hi]
    denominator = x2 - x1
    interpolated = ifelse(
        denominator == 0,
        y1,
        y1 + (y2 - y1) * (x - x1) / denominator,
    )
    first_value = @allowscalar fp[1]
    last_value = @allowscalar fp[n]
    return ifelse(i <= 0, first_value, ifelse(i >= n, last_value, interpolated))
end

"""
    searchsortedlast_vector(sorted_values, queries)

Pairwise binary search for a vector of queries. Unlike broadcasting
`searchsortedlast`, this lowers to logarithmically many one-dimensional gathers
instead of an `length(queries) × length(sorted_values)` comparison matrix.
"""
function searchsortedlast_vector(sorted_values, queries)
    n = length(sorted_values)
    low = zero.(round.(Int64, queries))
    high = low .+ (n + 1)
    iterations = ceil(Int, log2(n + 1))
    for _ in 1:iterations
        middle = (low .+ high) .÷ 2
        lookup = clamp.(middle, 1, n)
        candidates = sorted_values[lookup]
        move_low = (middle .<= n) .& (candidates .<= queries)
        low = ifelse.(move_low, middle, low)
        high = ifelse.(move_low, high, middle)
    end
    return low
end

function redshift_grid_index(redshift, z_grid)
    n = length(z_grid)
    i = searchsortedlast(z_grid, redshift)
    i_lo = ifelse(i <= 0, 1, ifelse(i >= n, n - 1, i))
    i_hi = i_lo + 1
    z_lo = @allowscalar z_grid[i_lo]
    z_hi = @allowscalar z_grid[i_hi]
    fraction = ifelse(z_hi == z_lo, 0.0, (redshift - z_lo) / (z_hi - z_lo))
    zero_based = (i_lo - 1) + fraction
    nearest = round(Int, zero_based) + 1
    return ifelse(i <= 0, 1, ifelse(i >= n, n, clamp(nearest, 1, n)))
end

"""
    generate_magnification(redshift, mag_tables, uniforms)

Use the same redshift-index interpolation and nearest-grid rounding as pySIDES.
The vectorized binary search avoids dynamic column slicing in Reactant.
"""
function generate_magnification(redshift, mag_tables, uniforms)
    (; z_grid, mu_grid, Psupmu) = mag_tables
    redshift_below = searchsortedlast_vector(z_grid, redshift)
    n_z = length(z_grid)
    z_lo_index = clamp.(redshift_below, 1, n_z - 1)
    z_hi_index = z_lo_index .+ 1
    z_lo = z_grid[z_lo_index]
    z_hi = z_grid[z_hi_index]
    z_denominator = z_hi .- z_lo
    z_safe_denominator = ifelse.(z_denominator .== 0.0, 1.0, z_denominator)
    z_fraction = (redshift .- z_lo) ./ z_safe_denominator
    z_fraction = ifelse.(z_denominator .== 0.0, 0.0, z_fraction)
    z_zero_based = (z_lo_index .- 1) .+ z_fraction
    z_nearest = round.(Int64, z_zero_based) .+ 1
    redshift_indices = ifelse.(
        redshift_below .<= 0,
        1,
        ifelse.(
            redshift_below .>= n_z,
            n_z,
            clamp.(z_nearest, 1, n_z),
        ),
    )
    n_mu = length(mu_grid)

    # Vectorized binary search across different CDF columns.  This performs
    # ceil(log2(n_mu + 1)) pairwise gathers instead of scanning every redshift
    # bin for every galaxy.
    low = zero.(round.(Int64, uniforms))
    high = low .+ (n_mu + 1)
    flat_cdf = vec(Psupmu)
    search_iterations = ceil(Int, log2(n_mu + 1))
    for _ in 1:search_iterations
        middle = (low .+ high) .÷ 2
        lookup_row = clamp.(middle, 1, n_mu)
        linear = lookup_row .+ (redshift_indices .- 1) .* n_mu
        values = flat_cdf[linear]
        move_low = (middle .<= n_mu) .& (values .<= uniforms)
        low = ifelse.(move_low, middle, low)
        high = ifelse.(move_low, high, middle)
    end

    i_lo = clamp.(low, 1, n_mu - 1)
    i_hi = i_lo .+ 1
    column_offset = (redshift_indices .- 1) .* n_mu
    x1 = flat_cdf[i_lo .+ column_offset]
    x2 = flat_cdf[i_hi .+ column_offset]
    y1 = mu_grid[i_lo]
    y2 = mu_grid[i_hi]
    denominator = x2 .- x1
    safe_denominator = ifelse.(denominator .== 0.0, 1.0, denominator)
    interpolated = y1 .+ (y2 .- y1) .* (uniforms .- x1) ./ safe_denominator
    interpolated = ifelse.(denominator .== 0.0, y1, interpolated)
    first_values = mu_grid[one.(i_lo)]
    last_values = mu_grid[i_lo .* 0 .+ n_mu]
    return ifelse.(
        low .<= 0,
        first_values,
        ifelse.(low .>= n_mu, last_values, interpolated),
    )
end

function sed_uindex(Umean, table)
    # Python uses round((Umean - Umean[0]) / dU), followed by zero-based
    # indexing.  The +1 conversion below is the exact Julia equivalent.
    u_min = @allowscalar table.Umean[1]
    index = round.(Int64, (Umean .- u_min) ./ table.dU) .+ 1
    return clamp.(index, 1, length(table.Umean))
end

function interpolate_sed_fluxes(lambda_list, sed_tables, redshift, Umean, issb)
    n_gal = length(redshift)
    n_lambda = length(lambda_list)
    uindex = sed_uindex(Umean, sed_tables)
    nuLnu_over_nu = zeros(eltype(redshift), n_gal, n_lambda)
    n_sed_lambda = length(sed_tables.lambda)
    ms_flat = vec(sed_tables.nuLnu_MS)
    sb_flat = vec(sed_tables.nuLnu_SB)
    column_offset = (uindex .- 1) .* n_sed_lambda

    # Work on one observed wavelength at a time. Pairwise flat gathers select
    # only the two bracketing SED values needed by each galaxy, avoiding an
    # n_gal × n_sed_lambda intermediate inside the compiled executable.
    for j in 1:n_lambda
        lambda_observed_um = @allowscalar lambda_list[j]
        lambda_rest_um = lambda_observed_um ./ (1.0 .+ redshift)
        indices = searchsortedlast_vector(sed_tables.lambda, lambda_rest_um)
        below = clamp.(indices, 1, n_sed_lambda - 1)
        above = below .+ 1
        below_linear = below .+ column_offset
        above_linear = above .+ column_offset

        x1 = sed_tables.lambda[below]
        x2 = sed_tables.lambda[above]
        denominator = x2 .- x1
        safe_denominator = ifelse.(denominator .== 0.0, 1.0, denominator)

        ms_below = ms_flat[below_linear]
        ms_above = ms_flat[above_linear]
        sb_below = sb_flat[below_linear]
        sb_above = sb_flat[above_linear]
        ms_interpolated = ms_below .+
            (ms_above .- ms_below) .* (lambda_rest_um .- x1) ./
            safe_denominator
        sb_interpolated = sb_below .+
            (sb_above .- sb_below) .* (lambda_rest_um .- x1) ./
            safe_denominator

        ms_value = ifelse.(
            indices .<= 0,
            ms_flat[one.(below_linear) .+ column_offset],
            ifelse.(
                indices .>= n_sed_lambda,
                ms_flat[n_sed_lambda .+ column_offset],
                ms_interpolated,
            ),
        )
        sb_value = ifelse.(
            indices .<= 0,
            sb_flat[one.(below_linear) .+ column_offset],
            ifelse.(
                indices .>= n_sed_lambda,
                sb_flat[n_sed_lambda .+ column_offset],
                sb_interpolated,
            ),
        )
        nu_rest_hz = C_M_PER_S ./ (lambda_rest_um .* 1.0e-6)
        @allowscalar nuLnu_over_nu[:, j] =
            ifelse.(issb, sb_value, ms_value) ./ nu_rest_hz
    end
    return nuLnu_over_nu
end

function generate_monochromatic_fluxes(
    lambda_list,
    sed_tables,
    redshift,
    magnified_lir,
    Umean,
    dlum_m,
    issb,
)
    shape = interpolate_sed_fluxes(
        lambda_list,
        sed_tables,
        redshift,
        Umean,
        issb,
    )
    lir_col = reshape(magnified_lir, :, 1)
    redshift_col = reshape(redshift, :, 1)
    dlum_col = reshape(dlum_m, :, 1)
    lnu = L_SUN_W .* lir_col .* shape
    return lnu .* (1.0 .+ redshift_col) ./
           (4.0 * pi .* dlum_col .^ 2) .* 1.0e26
end

function generate_lfir(ratio_tables, LIR, Umean, issb)
    uindex = sed_uindex(Umean, ratio_tables)
    ratio_ms = ratio_tables.LFIR_LIR_ratio_MS[uindex]
    ratio_sb = ratio_tables.LFIR_LIR_ratio_SB[uindex]
    return LIR .* ifelse.(issb, ratio_sb, ratio_ms)
end

function generate_dust_outputs(inputs, p, issb, mu, sfr, umean_normal)
    (; UmeanSB, UmeanMSz0, alphaMS, zlimMS, sigma_logUmean, SFR2LIR,
        lambda_list, sed_tables, ratio_tables) = p
    redshift = inputs.redshift
    zlim_sb = (log10(UmeanSB) - log10(UmeanMSz0)) / alphaMS
    zlim_sb = ifelse(zlim_sb > zlimMS, 9999.0, zlim_sb)

    ms_or_high_z = (.!issb) .| (redshift .>= zlim_sb)
    umean_ms = 10.0 .^
        (log10(UmeanMSz0) .+ alphaMS .* min.(redshift, zlimMS))
    umean_base = ifelse.(ms_or_high_z, umean_ms, UmeanSB)
    Umean = umean_base .* 10.0 .^ (sigma_logUmean .* umean_normal)
    LIR = SFR2LIR .* sfr

    monochromatic_fluxes = generate_monochromatic_fluxes(
        lambda_list,
        sed_tables,
        redshift,
        mu .* LIR,
        Umean,
        inputs.dlum_m,
        issb,
    )
    LFIR = generate_lfir(ratio_tables, LIR, Umean, issb)
    return (; Umean, LIR, monochromatic_fluxes, LFIR)
end

function filter_flux_one(redshift, Umean, LIR, mu, issb, qflag, table)
    n_u = length(table.Umean)
    n_z = table.Nlog1plusz
    u_index = clamp.(round.(Int64, Umean ./ table.dU), 1, n_u)
    z_float = log10.(1.0 .+ redshift) ./ table.dlog1plusz .- 1.0
    below_raw = floor.(Int64, z_float) .+ 1
    above_raw = ceil.(Int64, z_float) .+ 1
    below = clamp.(below_raw, 1, n_z)
    above = clamp.(above_raw, 1, n_z)

    z_below = 10.0 .^ (table.dlog1plusz .* below) .- 1.0
    z_above = 10.0 .^ (table.dlog1plusz .* above) .- 1.0
    denominator = z_above .- z_below
    exact = denominator .== 0.0
    safe_denominator = ifelse.(exact, 1.0, denominator)
    weight_below = ifelse.(
        exact,
        0.5,
        (z_above .- redshift) ./ safe_denominator,
    )
    weight_above = ifelse.(
        exact,
        0.5,
        (redshift .- z_below) ./ safe_denominator,
    )

    too_low = below_raw .< 1
    too_high = above_raw .> n_z
    z_min = 10.0 ^ table.dlog1plusz - 1.0
    low_scale = (z_min ./ max.(redshift, eps(Float64))) .^ 2
    weight_below = ifelse.(too_low, 0.0, ifelse.(too_high, 1.0, weight_below))
    weight_above = ifelse.(
        too_low,
        low_scale,
        ifelse.(too_high, 0.0, weight_above),
    )

    # Convert pairwise (U, z) coordinates to column-major linear indices.
    # One-dimensional traced gather is supported by Reactant and avoids
    # CartesianIndex construction on traced arrays.
    below_linear = u_index .+ (below .- 1) .* n_u
    above_linear = u_index .+ (above .- 1) .* n_u
    ms_flat = vec(table.Snu_LIR_MS)
    sb_flat = vec(table.Snu_LIR_SB)
    ms_value = ms_flat[below_linear] .* weight_below .+
               ms_flat[above_linear] .* weight_above
    sb_value = sb_flat[below_linear] .* weight_below .+
               sb_flat[above_linear] .* weight_above
    template_value = ifelse.(issb, sb_value, ifelse.(qflag, 0.0, ms_value))
    return template_value .* LIR .* mu
end

function generate_filter_fluxes(redshift, Umean, LIR, mu, issb, qflag, filter_tables)
    n_gal = length(redshift)
    n_filter = length(filter_tables)
    n_filter == 0 && return nothing
    result = zeros(eltype(redshift), n_gal, n_filter)
    for filter_index in 1:n_filter
        values = filter_flux_one(
            redshift,
            Umean,
            LIR,
            mu,
            issb,
            qflag,
            filter_tables[filter_index],
        )
        @allowscalar result[:, filter_index] = values
    end
    return result
end

function generate_line_outputs(inputs, sfr, qflag, issb, mu, Umean, LIR, p, noise)
    redshift = inputs.redshift
    dlum_mpc = inputs.dlum_mpc
    n_gal = length(redshift)

    safe_sfr = max.(sfr, floatmin(Float64))
    lprim_base = 10.0 .^ (0.81 .* (log10.(safe_sfr) .+ 10.0) .+ 0.54)
    lprim_base = ifelse.(qflag, 0.0, lprim_base)
    LprimCO10 = lprim_base .* ifelse.(issb, 10.0 ^ (-0.46), 1.0) .*
                  10.0 .^ (p.sigma_dex_CO10 .* noise.co_normal)
    nu_co_obs = p.nu_CO ./ (1.0 .+ redshift)
    ICO10 = mu .* LprimCO10 .* (1.0 .+ redshift) .^ 3 .* nu_co_obs .^ 2 ./
            (dlum_mpc .^ 2 .* 3.25e7)

    ratio_54_21 = 10.0 .^ (0.6 .* log10.(Umean) .- 0.38)
    diffuse_21 = @allowscalar p.Idiffuse[2]
    diffuse_54 = @allowscalar p.Idiffuse[5]
    clump_21 = @allowscalar p.Iclump[2]
    clump_54 = @allowscalar p.Iclump[5]
    fclump = (
        diffuse_54 .- ratio_54_21 .* diffuse_21
    ) ./ (
        ratio_54_21 .* (clump_21 - diffuse_21) .-
        (clump_54 - diffuse_54)
    )
    fclump = clamp.(fclump, 0.0, 1.0)

    co_fluxes = zeros(eltype(redshift), n_gal, 7)
    @trace track_numbers=false for k in 1:7
        diffuse_ratio = @allowscalar p.Idiffuse[k + 1]
        clump_ratio = @allowscalar p.Iclump[k + 1]
        birkin_ratio = @allowscalar p.rJup1_Birkin[k]
        ms_ratio = fclump .* clump_ratio .+ (1.0 .- fclump) .* diffuse_ratio
        sb_ratio = birkin_ratio * (k + 1) ^ 2
        transition_ratio = if p.SLED_SB_Birkin
            ifelse.(issb, sb_ratio, ifelse.(qflag, 0.0, ms_ratio))
        else
            ms_ratio
        end
        @allowscalar co_fluxes[:, k] = ICO10 .* transition_ratio
    end

    LCII_Lagache_raw = safe_sfr .^ (1.4 .- 0.07 .* redshift) .*
                        10.0 .^ (7.1 .- 0.07 .* redshift) .*
                        10.0 .^ (p.sigma_dex_CII .* noise.cii_lagache_normal)
    LCII_Lagache = ifelse.(sfr .> 0.0, LCII_Lagache_raw, 0.0)
    ICII_Lagache = mu .* LCII_Lagache ./
                   (1.04e-3 .* dlum_mpc .^ 2 .* (p.nu_CII ./ (1.0 .+ redshift)))
    LCII_de_Looze = 10.0 ^ 7.06 .* sfr .*
                    10.0 .^ (p.sigma_dex_CII .* noise.cii_delooze_normal)
    ICII_de_Looze = mu .* LCII_de_Looze ./
                    (1.04e-3 .* dlum_mpc .^ 2 .* (p.nu_CII ./ (1.0 .+ redshift)))

    ICO43 = @allowscalar co_fluxes[:, 3]
    ICO76 = @allowscalar co_fluxes[:, 6]
    has_lir = LIR .> 0.0
    safe_lir = max.(LIR, floatmin(Float64))
    safe_ico43 = max.(ICO43, floatmin(Float64))
    safe_ico76 = max.(ICO76, floatmin(Float64))
    lco43_lir = 1.04e-3 .* safe_ico43 .* dlum_mpc .^ 2 .* (4.0 * p.nu_CO) ./
                ((1.0 .+ redshift) .* safe_lir)
    LCI10 = 10.0 .^ (p.a_CI10 .* log10.(lco43_lir) .+ p.b_CI10) .* LIR .*
            10.0 .^ (p.sigma_CI10 .* noise.ci10_normal)
    ICI10_raw = LCI10 .* (1.0 .+ redshift) ./
                (1.04e-3 .* dlum_mpc .^ 2 .* p.nu_CI10)
    ICI10 = ifelse.(has_lir, ICI10_raw, 0.0)

    co76_co43 = safe_ico76 ./ safe_ico43 .* (7.0 / 4.0)
    LCI21 = 10.0 .^ (p.a_CI21 .* log10.(co76_co43) .+ p.b_CI21) .* LCI10 .*
            10.0 .^ (p.sigma_CI21 .* noise.ci21_normal)
    ICI21_raw = LCI21 .* (1.0 .+ redshift) ./
                (1.04e-3 .* dlum_mpc .^ 2 .* p.nu_CI21)
    ICI21 = ifelse.(has_lir, ICI21_raw, 0.0)

    return (;
        LprimCO10,
        ICO10,
        co_fluxes,
        LCII_Lagache,
        ICII_Lagache,
        LCII_de_Looze,
        ICII_de_Looze,
        ICI10,
        ICI21,
    )
end

"""
    forward_model(inputs, params, noise)

Pure numerical forward model shared by the Julia and Reactant paths.  `inputs`,
`params`, and `noise` contain only numbers, arrays, named tuples, and tuples, so
all file parsing and catalog I/O stays outside Reactant compilation.
"""
function forward_model(inputs, params, noise)
    sfr = generate_sfr(inputs.redshift, inputs.Mstar, params.sfr, noise)
    mu = generate_magnification(inputs.redshift, params.magnification, noise.mu_uniform)
    dust = generate_dust_outputs(
        inputs,
        params.dust,
        sfr.issb,
        mu,
        sfr.SFR,
        noise.umean_normal,
    )
    filter_fluxes = generate_filter_fluxes(
        inputs.redshift,
        dust.Umean,
        dust.LIR,
        mu,
        sfr.issb,
        sfr.qflag,
        params.filters,
    )
    lines = generate_line_outputs(
        inputs,
        sfr.SFR,
        sfr.qflag,
        sfr.issb,
        mu,
        dust.Umean,
        dust.LIR,
        params.lines,
        noise,
    )
    return (; sfr..., mu, dust..., filter_fluxes, lines...)
end

function forward_model_impulse(rng, inputs, base_params, noise)

    _, m0 = ProbProg.sample(
        rng, ProbProg.Normal(0.0, 2.0, (1,)); symbol=:m0
    )

    _, a0 = ProbProg.sample(
        rng, ProbProg.Normal(0.0, 2.0, (1,)); symbol=:a0
    )

    _, sigma_MS = ProbProg.sample(
        rng, ProbProg.Normal(0.0, 1.0, (1,)); symbol=:sigma_MS
    )

    sfr_params = merge(
        base_params.sfr,
        (; m0, a0, sigma_MS)
    )

    params = merge(base_params, (; sfr=sfr_params))

    return forward_model(inputs, params, noise)
end
