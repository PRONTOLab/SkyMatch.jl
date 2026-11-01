using Test
using SkyMatch
using DataFrames: DataFrame
using FITSIO: FITS, read_header
using HDF5: h5open
using WCS: WCS, world_to_pix

function read_map_fixture(path)
    FITS(path, "r") do file
        return read(file[1]), read_header(file[1])
    end
end

@testset "Missing filter fluxes and shared generated coordinates" begin
    # Constant filter tables give a closed-form flux independent of interpolation:
    # MS = 2*mu*LIR, SB = 5*mu*LIR, and passive sources have zero flux.
    cat = DataFrame(
        redshift=[0.3, 0.4, 0.5], Umean=ones(3), LIR=[2.0, 3.0, 4.0],
        mu=[1.0, 2.0, 1.0], issb=[false, true, false], qflag=[false, false, true],
    )
    mktempdir() do dir
        for (name, factor) in (("alpha", 1.0), ("beta", 2.0))
            h5open(joinpath(dir, "$name.h5"), "w") do file
                file["dU"] = 1.0
                file["dlog1plusz"] = 0.1
                file["Nlog1plusz"] = 3
                file["Umean"] = [1.0, 2.0]
                file["Snu_LIR_MS"] = fill(2.0 * factor, 2, 3)
                file["Snu_LIR_SB"] = fill(5.0 * factor, 2, 3)
            end
        end
        params = Dict{String,Any}(
            "filter_list" => ["alpha", "beta"],
            "pixel_size" => [1.0, 1.0], "beam_fwhm_list" => [2.0, 2.0],
            "field_size" => 1.0e-5,
            "output_path" => dir, "run_name" => "fallback",
            "sides_cat_path" => "synthetic", "gen_map_nobeam_Jy_pix" => true,
        )
        result = SkyMatch.make_maps(cat, params, Dict{String,Any}("grid_filter_path" => dir))
        @test result.Salpha ≈ [4.0, 30.0, 0.0]
        @test result.Sbeta ≈ [8.0, 60.0, 0.0]
        @test all(isfinite, result.ra)
        @test all(isfinite, result.dec)
        alpha, alpha_header = read_map_fixture(joinpath(dir, "fallback_alpha_nobeam_Jy_pix.fits"))
        beta, beta_header = read_map_fixture(joinpath(dir, "fallback_beta_nobeam_Jy_pix.fits"))
        @test sum(alpha) ≈ 34.0
        @test beta ≈ 2 .* alpha
        for key in ("CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "CDELT1", "CDELT2")
            @test alpha_header[key] == beta_header[key]
        end
    end
end

@testset "Invalid map geometry" begin
    cat = DataFrame(ra=[10.0], dec=[0.0], Stest=[1.0])
    @test_throws ArgumentError SkyMatch.set_wcs_map(cat[1:0, :], 1.0, Dict{String,Any}())
    for pixel_size in (0.0, -1.0, NaN, Inf)
        @test_throws ArgumentError SkyMatch.set_wcs_map(cat, pixel_size, Dict{String,Any}())
    end
    mktempdir() do dir
        params = Dict{String,Any}(
            "filter_list" => ["test"], "pixel_size" => [1.0, 2.0],
            "beam_fwhm_list" => [2.0], "output_path" => dir,
        )
        @test_throws DimensionMismatch SkyMatch.make_maps(cat, params, Dict{String,Any}())
    end
end

@testset "Map binning and FITS coordinates" begin
    # Deliberately non-unit bins: the supplied edges define the histogram.
    hist = SkyMatch.compute_histogram2d(
        [10.5, 12.5, 10.5], [21.0, 25.0, 21.0],
        [10.0, 12.0, 14.0], [20.0, 24.0, 28.0], [2.0, 3.0, 5.0],
    )
    @test hist == [7.0 0.0; 0.0 3.0]
    @test sum(hist) == 10.0

    cat = DataFrame(
        ra=[10.0, 10.004, 10.0013],
        dec=[-0.002, 0.0, -0.0009],
        Stest=[2.0, 3.0, 5.0],
    )
    props = SkyMatch.set_wcs_map(cat, 1.0, Dict{String,Any}())
    pixels = world_to_pix(props["w"], permutedims(hcat(cat.ra, cat.dec)))
    hist = SkyMatch.compute_histogram2d(
        props["pos"][1], props["pos"][2],
        props["y_edges"], props["x_edges"], cat.Stest,
    )
    @test sum(hist) ≈ sum(cat.Stest)
    @test size(hist, 1) != size(hist, 2)

    mktempdir() do dir
        path = joinpath(dir, "coordinates.fits")
        SkyMatch.save_map(path, hist, props, "test", "Jy/pix", 0.0, "synthetic")
        image, header = read_map_fixture(path)
        # FITS axis 1 is RA/x, whereas the internal histogram is (y,x).
        @test size(image) == reverse(size(hist))
        @test header["CTYPE1"] == "RA---TAN"
        @test header["CTYPE2"] == "DEC--TAN"
        @test header["CRPIX1"] ≈ props["w"].crpix[1]
        @test header["CRPIX2"] ≈ props["w"].crpix[2]
        @test header["BUNIT"] == "Jy/pix"
        file_wcs = FITS(path, "r") do file
            only(WCS.from_header(read_header(file[1], String)))
        end
        file_pixels = world_to_pix(file_wcs, permutedims(hcat(cat.ra, cat.dec)))
        @test file_pixels ≈ pixels
        for source in eachindex(cat.ra)
            ix, iy = round.(Int, file_pixels[:, source])
            @test 1 <= ix <= size(image, 1)
            @test 1 <= iy <= size(image, 2)
            @test image[ix, iy] ≈ cat.Stest[source]
        end
    end
end

@testset "Rectangular beam maps and brightness units" begin
    # Zero-flux sources set a rectangular footprint around one isolated source.
    cat = DataFrame(
        ra=[10.0, 10.004, 10.0013],
        dec=[-0.002, 0.0, -0.0009],
        Stest=[0.0, 0.0, 3.5],
    )
    pixel_size = 1.0
    pixel_sr = (pixel_size * pi / (180 * 3600))^2
    mktempdir() do dir
        params = Dict{String,Any}(
            "filter_list" => ["test"],
            "pixel_size" => [pixel_size],
            "beam_fwhm_list" => [2.0],
            "output_path" => dir,
            "run_name" => "regression",
            "sides_cat_path" => "synthetic",
            "gen_map_nobeam_Jy_pix" => true,
            "gen_map_nobeam_MJy_sr" => true,
            "gen_map_smoothed_Jy_beam" => true,
            "gen_map_smoothed_MJy_sr" => true,
        )
        SkyMatch.make_maps(cat, params, Dict{String,Any}())
        load_product(suffix) = read_map_fixture(joinpath(dir, "regression_test_$(suffix).fits"))
        raw, raw_header = load_product("nobeam_Jy_pix")
        brightness, brightness_header = load_product("nobeam_MJy_sr")
        beam, beam_header = load_product("smoothed_Jy_beam")
        beam_brightness, beam_brightness_header = load_product("smoothed_MJy_sr")

        @test size(raw, 1) != size(raw, 2)
        @test size(raw) == size(beam) == size(brightness) == size(beam_brightness)
        @test all(isfinite, beam)
        @test sum(raw) ≈ 3.5
        @test sum(brightness) * pixel_sr * 1e6 ≈ 3.5
        @test sum(beam_brightness) * pixel_sr * 1e6 ≈ 3.5
        @test argmax(beam) == argmax(raw)
        @test maximum(beam) ≈ 3.5
        @test brightness_header["BUNIT"] == "MJy/sr"
        @test beam_header["BUNIT"] == "Jy/beam"
        @test beam_brightness_header["BUNIT"] == "MJy/sr"
    end
end
