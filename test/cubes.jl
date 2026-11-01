using Test
using SkyMatch
using DataFrames: DataFrame
using FITSIO
using HDF5: h5open
using Unitful
using UnitfulAstro
using WCS

@testset "Continuum and spectral cubes" begin
    sed = Dict{String,Any}(
        "lambda" => [1.0, 10000.0],
        "Umean" => [1.0, 2.0, 3.0],
        "dU" => 1.0,
        "nuLnu_MS_arr" => [1.0 4.0 9.0; 1.0 4.0 9.0],
        "nuLnu_SB_arr" => [2.0 8.0 18.0; 2.0 8.0 18.0],
    )
    wavelengths = [100.0, 200.0, 300.0]
    z = [0.0, 0.01]
    luminosity = [1.0e10, 2.0e10]
    distance = [100.0, 200.0] # Mpc, as in add_output_columns!
    flux = SkyMatch.gen_Snu_arr(wavelengths, sed, z, luminosity, [2.0, 3.0], distance, [false, true])
    @test size(flux) == (2, 3)
    for i in 1:2, j in 1:3
        shape = (i == 1 ? 4.0 : 18.0) / (SkyMatch.C_M_PER_S / (wavelengths[j] * 1e-6 / (1 + z[i])))
        expected = SkyMatch.L_SUN_W * luminosity[i] * shape * (1 + z[i]) /
                   (4pi * (distance[i] * SkyMatch.MPC_TO_M)^2) * 1e26
        @test flux[i, j] ≈ expected
    end
    @test SkyMatch.gen_Snu_arr(wavelengths, sed, z, luminosity, [2.0, 3.0], distance .* u"Mpc", [false, true]) ≈ flux

    mktempdir() do directory
        sed_path = joinpath(directory, "sed.h5")
        h5open(sed_path, "w") do file
            for (name, values) in sed
                file[name] = values
            end
        end
        cat = DataFrame(ra=[0.0, 4 / 3600], dec=[0.0, 2 / 3600],
                        redshift=z, mu=ones(2), LIR=luminosity, Umean=[2.0, 3.0],
                        Dlum=distance, issb=[false, true], ICO10=[3.0, 5.0])
        params = Dict{String,Any}(
            "pixel_size" => 1.0, "freq_min" => 100.0e9,
            "freq_max" => 101.0e9, "freq_resol" => 1.0e9,
            "telescop_diameter" => 1000.0,
            "output_path" => directory, "run_name" => "tiny",
            "sides_cat_path" => "synthetic", "save_continuum_only" => false,
            "save_cube_nobeam_Jy_pix" => true,
            "gen_cube_smoothed_Jy_beam" => true,
            "gen_cube_smoothed_MJy_sr" => true,
        )
        sides = Dict{String,Any}("SED_file" => sed_path, "nu_CO" => 101.0,
                                "nu_CI10" => 101.0, "nu_CI21" => 101.0,
                                "nu_CII" => 101.0)
        props = SkyMatch.set_wcs(cat, params)
        @test props["shape"][1] == 2 # An even channel count must stay even.
        @test props["shape"][2] != props["shape"][3]
        kernels, areas = SkyMatch.set_kernel(params, props)
        @test all(size(kernel) == Tuple(props["shape"][2:3]) for kernel in kernels)
        props["kernel"], props["beam_area_pix2"] = kernels, areas

        continuum = SkyMatch.make_continuum_cube(cat, sides, params, props)
        channel_flux = SkyMatch.channel_flux_densities(cat, sides, params)
        @test size(continuum["nobeam_Jy_pix"]) == Tuple(props["shape"])
        @test vec(sum(continuum["nobeam_Jy_pix"]; dims=(2, 3))) ≈ vec(sum(channel_flux; dims=1))

        line_flux, channels = SkyMatch.line_channel_flux_densities("CO10", 101.0, cat, props)
        @test channels ≈ [1.0, 0.0] atol=1e-12 # Highest and lowest channel centers.
        line_cube = SkyMatch.compute_histogram3d(channels, props["pos"][1], props["pos"][2],
            props["z_edges"], props["y_edges"], props["x_edges"], line_flux)
        @test vec(sum(line_cube; dims=(2, 3))) ≈ reverse(line_flux)
        @test sum(line_cube) ≈ sum(line_flux)

        impulse = zeros(Float64, props["shape"]...)
        impulse[1, 2, 2] = 1.0
        outputs = SkyMatch.save_cubes(impulse, props, sides, params, "impulse"; just_compute=true)
        @test argmax(outputs["smoothed_Jy_beam"]) == CartesianIndex(1, 2, 2)
        pixel_sr = (params["pixel_size"] * pi / 180 / 3600)^2
        @test sum(outputs["smoothed_MJy_sr"]) * pixel_sr * 1e6 ≈ 1.0
        @test sum(outputs["smoothed_Jy_beam"]) ≈ areas[1]
        SkyMatch.save_cubes(outputs, props, sides, params, "impulse"; just_save=true)
        FITS(joinpath(directory, "tiny_impulse_nobeam_Jy_pix.fits")) do file
            @test read(file[1]) == permutedims(impulse, (3, 2, 1))
            header = read_header(file[1])
            @test header["CTYPE1"] == "RA---TAN"
            @test header["CTYPE2"] == "DEC--TAN"
            @test header["CTYPE3"] == "FREQ"
            @test header["CRPIX3"] == 1.0
            @test header["CRVAL3"] == params["freq_min"]
            restored_wcs = only(WCS.from_header(read_header(file[1], String)))
            @test WCS.pix_to_world(restored_wcs, [2.0, 2.0, 1.0]) ≈
                  WCS.pix_to_world(props["w"], [2.0, 2.0, 1.0])
        end

        for j in 2:8
            cat[!, Symbol("ICO$(j)$(j - 1)")] = zeros(2)
        end
        cat.ICI10 = zeros(2)
        cat.ICI21 = zeros(2)
        cat.ICII_Lagache = zeros(2)
        params["gen_cube_CII_Lagache"] = true
        params["save_full"] = true
        @test SkyMatch.make_cube(cat, sides, params)
        @test isfile(joinpath(directory, "tiny_full_Lagache_nobeam_Jy_pix.fits"))
    end
    @test_throws ArgumentError SkyMatch.cube_frequencies(Dict("freq_min" => 1.0, "freq_max" => 2.0, "freq_resol" => 0.0))
end
