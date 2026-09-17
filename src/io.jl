using Base.Meta  # Needed for Meta.parse (optional, as Base.Meta exports Meta.parse)
using DataFrames
using CSV
using FITSIO
using WCS
using Unitful
using UnitfulAstro
using Serialization
using Statistics
using Dates
using FFTW
using Interpolations
using HDF5
using Distributed

# Constants to load
const gaussian_fwhm_to_sigma = 1.0 / (2.0 * sqrt(2.0 * log(2.0)))
const c_speed = 299792458.0 # m/s

"""
    load_par_file(path::AbstractString, force_pysides_path::AbstractString="")

Reads a configuration file, ignoring comments starting with '#', and parses 
the values dynamically using Meta.parse and Base.eval, similar to Python's eval.
Returns a Dict{String, Any}.
"""
function load_par_file(path::AbstractString, force_pysides_path::AbstractString="")
    params = Dict{String,Any}()

    # 1. Read and Parse Key-Value Pairs
    # Use the 'do' block syntax for safe file handling (ensures 'file' is closed)
    open(path, "r") do file
        for line in eachline(file)
            line_stripped = strip(line)
            # Skip lines starting with '#' (comments)
            if !startswith(line_stripped, "#")
                # Split line by the first occurrence of '#', keeping only the key/value part
                no_comment = split(line_stripped, '#', limit=2)[1]
                # Split key and value by the first occurrence of '='
                key_value = split(no_comment, '=', limit=2)
                if length(key_value) == 2
                    key = strip(key_value[1])
                    value_str = strip(key_value[2])
                    params[key] = value_str
                end
            end
        end
    end

    # 2. Evaluate and Convert Types (Equivalent to Python's eval loop)
    # This dynamic evaluation is necessary to convert strings like "3.14" 
    # or "[1-3]" into their corresponding Julia types.
    for (key, value_str) in params
        try
            ex = Meta.parse(value_str)
            params[key] = Base.eval(Main, ex)
        catch e
            @warn "Could not evaluate parameter '$key'. Keeping value as raw string: $value_str" exception = (e, catch_backtrace())
        end
    end
    return params
end

"""
    load_sides_csv(catfile::AbstractString, nrows::Union{Nothing, Int} = nothing)

Loads a catalog CSV, selecting the primary columns required for analysis.
nrows: Limits the number of rows read from the file (excluding headers).
"""
function load_sides_csv(catfile::AbstractString, nrows::Union{Nothing,Int}=nothing)

    # Equivalent to Python's print()
    println("Load the catalog CSV generated from the original IDL code to get RA, Dec, z, Mhalo, and Mstar...") # [9]

    # Equivalent to pd.read_csv, reading directly into a DataFrame [8].
    # The 'limit' keyword argument handles the Python 'nrows' parameter [11].
    # We explicitly specify the delimiter as ',' using 'delim' (equivalent to 'sep=', assuming a standard CSV).
    cat_IDL = CSV.read(
        catfile,
        DataFrame;
        delim=',',
        limit=nrows # Reads only up to `nrows`
    )

    # --- Column Selection and Renaming (Equivalent to pd.DataFrame(..., columns=...)) ---

    # Define the list of required columns (as Symbols in Julia, which represent column names)
    required_cols = [:redshift, :ra, :dec, :Mhalo, :Mstar]

    # Select only the required columns and return a new DataFrame [2, 12].
    # This assumes the CSV file already contains columns with these exact names.
    cat = cat_IDL[!, required_cols]

    return cat
end

function gen_outputs(cat::DataFrame, params::Dict)
    # 1. Handle directory creation
    output_path = params["output_path"]
    if !isdir(output_path)
        println("Create $output_path")
        mkpath(output_path) # Equivalent to os.makedirs
    end

    # 2. Export to Pickle (Serialization)
    # In Julia, serialize is the standard way to dump objects to a binary file
    #if get(params, "gen_pickle", false) == true
    #    file_p = joinpath(output_path, params["run_name"] * ".p")
    #    println("Export the catalog to pickle... ($file_p)")
    #    open(file_p, "w") do f
    #        serialize(f, cat)
    #    end
    #end

    # 3. Export to FITS
    if get(params, "gen_fits", false) == true
        file_fits = joinpath(output_path, params["run_name"] * ".fits")
        println("Export the catalog to FITS... ($file_fits)")

        # Create a copy to add units without mutating the original catalog
        export_cat = copy(cat)
        col_names = names(export_cat)

        # Create a compatible dictionary for FITSIO
        data_dict = Dict{String,AbstractVector}()

        for name in col_names
            # 1. Extract the column and strip any physical units 
            col = ustrip.(export_cat[!, name])

            # 2. Check for BitVector and convert to standard Vector{Bool}
            # Standard Arrays are recognized by FITSIO's Array{T} method 
            if col isa BitVector
                col = Vector{Bool}(col)
            end

            data_dict[name] = col
        end

        # 1. Prepare three vectors to define the header records
        # FITS records consist of a Key, a Value, and a Comment
        h_keys = String[]
        h_vals = Any[]
        h_comms = String[]

        # 2. Populate the vectors with your simulation parameters
        for (key, val) in params
            # We use "COMMENT" as the keyword for every parameter
            push!(h_keys, "COMMENT")
            # COMMENT cards have no value in the standard FITS format
            push!(h_vals, nothing)
            # Store the "Key = Value" string in the comment field of the record
            push!(h_comms, "$key = $val")
        end
        header = FITSHeader(h_keys, h_vals, h_comms)

        # 2. Add your parameters as "COMMENT" cards to the header object
        # FITS supports multiple entries under the "COMMENT" keyword
        #for (key, val) in params
        #    # set_comment! adds or modifies metadata in a header object [1]
        #    # We use "COMMENT" as the key to create standard FITS comment lines
        #    set_comment!(header, "COMMENT", "$key = $val")
        #end

        # 3. Write to FITS using the function-block syntax to ensure the file closes
        FITS(file_fits, "w") do f
            # This will now succeed because all types are standard arrays
            write(f, data_dict; header=header)

            ## 4. Add simulation parameters as comments
            #for (key, val) in params
            #    FITSIO.write_key(f[1], "$key = $val")
            #end
        end
    end

    return true
end

function compute_histogram2d(y::AbstractVector, x::AbstractVector, y_edges::AbstractVector, x_edges::AbstractVector, weights::AbstractVector)
    ny = length(y_edges) - 1
    nx = length(x_edges) - 1
    histo = zeros(Float64, ny, nx)
    
    for i in eachindex(x)
        iy = floor(Int, y[i] + 0.5) + 1
        ix = floor(Int, x[i] + 0.5) + 1
        
        if 1 <= iy <= ny && 1 <= ix <= nx
            histo[iy, ix] += weights[i]
        end
    end
    return histo
end

function compute_histogram3d(z::AbstractVector, y::AbstractVector, x::AbstractVector, 
                             z_edges::AbstractVector, y_edges::AbstractVector, x_edges::AbstractVector, 
                             weights::AbstractVector)
    nz = length(z_edges) - 1
    ny = length(y_edges) - 1
    nx = length(x_edges) - 1
    histo = zeros(Float64, nz, ny, nx)
    
    for i in eachindex(x)
        iz = floor(Int, z[i] + 0.5) + 1
        iy = floor(Int, y[i] + 0.5) + 1
        ix = floor(Int, x[i] + 0.5) + 1
        
        if 1 <= iz <= nz && 1 <= iy <= ny && 1 <= ix <= nx
            histo[iz, iy, ix] += weights[i]
        end
    end
    return histo
end

# SED and flux interpolation utilities


function grouper(N::Int, n::Int)
    if n <= 0
        throw(ArgumentError("Number of chunks (n) must be positive"))
    end
    N_per_chunk = ceil(Int, N / n)
    chunks = []
    start_idx = 1
    while start_idx <= N
        end_idx = min(N, start_idx + N_per_chunk - 1)
        push!(chunks, start_idx:end_idx)
        start_idx = end_idx + 1
    end
    return [chunk for chunk in chunks if !isempty(chunk)]
end

function worker(ks, lambda_list, stype, Uindex, SED_dict, redshift)
    ks = filter(!isnothing, ks)
    N_gal = length(ks)
    N_lambda = length(lambda_list)
    nuLnu = zeros(Float64, N_gal, N_lambda)

    redshift_subset = redshift[ks]
    redshift_col = reshape(redshift_subset, N_gal, 1)
    lambda_rest = (lambda_list ./ (1.0 .+ redshift_col)) .* u"μm"

    c_speed_q = uconvert(u"m/s", c_speed * u"m/s")
    nu_rest_Hz = (c_speed_q ./ (lambda_rest .|> u"m")) .|> u"Hz"

    for i in 1:N_gal
        k = ks[i]
        lambda_interp_x = SED_dict["lambda"]
        sed_data = SED_dict[stype[k]][:, Uindex[k]]

        interp_itp = LinearInterpolation(lambda_interp_x, sed_data, extrapolation_bc=Line())
        lambda_rest_row = lambda_rest[i, :] .|> Unitful.NoUnits
        nuLnu_row = interp_itp.(lambda_rest_row)
        nuLnu[i, :] = nuLnu_row
    end

    nuLnu_quantified = nuLnu .* u"W"
    return (nuLnu_quantified ./ nu_rest_Hz) .|> Unitful.NoUnits
end

function gen_Snu_arr(lambda_list, SED_dict, redshift, LIR, Umean, Dlum, issb)
    N_total = length(redshift)
    stype = [a ? "nuLnu_SB_arr" : "nuLnu_MS_arr" for a in issb]

    Umean_min = SED_dict["Umean"][1]
    dU = SED_dict["dU"]

    Uindex = round.((Umean .- Umean_min) ./ dU)
    Uindex = Int.(Uindex)

    Umax_index = length(SED_dict["Umean"])
    Uindex = max.(Uindex, 1)
    Uindex = min.(Uindex, Umax_index)

    CPU_COUNT = Sys.CPU_THREADS
    index_chunks = grouper(N_total, CPU_COUNT)

    Worker_partial(ks) = worker(
        ks,
        lambda_list,
        stype,
        Uindex,
        SED_dict,
        redshift
    )

    L_nu_over_nu_chunks = nprocs() > 1 ? pmap(Worker_partial, index_chunks) : map(Worker_partial, index_chunks)
    concatenated_worker_output = vcat(L_nu_over_nu_chunks...)

    L_sun_W = 3.828e26 * u"W"
    LIR_col = reshape(LIR, N_total, 1)
    Lnu = L_sun_W .* LIR_col .* concatenated_worker_output ./ u"Hz"

    redshift_col = reshape(redshift, N_total, 1)
    Numerator = Lnu .* (1.0 .+ redshift_col) .* (1.0 / (pi * 4.0))

    Dlum_m_squared = (Dlum .|> u"m") .^ 2
    Denominator = reshape(Dlum_m_squared, N_total, 1)

    Snu_arr = (Numerator ./ Denominator) .|> u"Jy"
    return Snu_arr
end

function load_sed_pickle_equivalent(file_path::String)
    println("Loading data from hdf5 file: $file_path")
    SEDData = Dict{String, Any}()
    try
        h5open(file_path, "r") do f
            for key in keys(f)
                SEDData[key] = read(f[key])
            end
        end
        println("Successfully loaded SEDData from HDF5!")
    catch e
        println("Error loading HDF5 file: ", e)
    end
    return SEDData
end


# World coordinate system (WCS) and kernel computations


function gen_radec(cat::DataFrame, params::Dict)
    n = nrow(cat)
    ra_max = sqrt(params["field_size"])
    dec_max = sqrt(params["field_size"])

    ra = rand(n) .* ra_max .* u"°"
    dec = rand(n) .* dec_max .* u"°"
    return ra, dec 
end

function set_wcs_map(cat::DataFrame, pixel_size::Real, params::Dict)
    local ra, dec
    if !("ra" in names(cat)) || !("dec" in names(cat))
        println("generating the coordinates of the sources")
        ra, dec = gen_radec(cat, params)
    else
        ra = cat.ra .* u"°"
        dec = cat.dec .* u"°"
    end
    
    ra_val = ustrip.(u"°", ra)
    dec_val = ustrip.(u"°", dec)
    
    pix_resol = pixel_size / 3600.
    
    ra_cen = 0.5 * (maximum(ra_val) + minimum(ra_val))
    dec_cen = 0.5 * (maximum(dec_val) + minimum(dec_val))
    delta_ra = maximum(ra_val) - minimum(ra_val)
    delta_dec = maximum(dec_val) - minimum(dec_val)
    
    w = WCSTransform(2)
    w.crval = [ra_cen, dec_cen]
    w.crpix = [0.5 * delta_ra / pix_resol, 0.5 * delta_dec / pix_resol]
    w.cdelt = [pix_resol, pix_resol]
    w.ctype = ["RA---TAN", "DEC--TAN"]
    w.cunit = ["deg", "deg"]
    
    worldcoords = vcat(ra_val', dec_val')
    pixcoords = world_to_pix(w, worldcoords)
    x = pixcoords[1, :]
    y = pixcoords[2, :]
    
    w.crpix = [
        0.5 * delta_ra / pix_resol - minimum(x),
        0.5 * delta_dec / pix_resol - minimum(y)
    ]
    
    pixcoords = world_to_pix(w, worldcoords)
    x = pixcoords[1, :]
    y = pixcoords[2, :]
    
    pos = [y, x]
    
    shape_val = [Int(ceil(maximum(y))), Int(ceil(maximum(x)))]
    shape = [div(i, 2) * 2 + 1 for i in shape_val]
    
    x_edges = collect(-0.5:1.0:(shape[2]-0.5))
    y_edges = collect(-0.5:1.0:(shape[1]-0.5))
    
    wcs_dict = Dict{String, Any}(
        "w" => w,
        "shape" => shape,
        "pos" => pos,
        "x_edges" => x_edges,
        "y_edges" => y_edges
    )
    return wcs_dict
end

function set_wcs(cat::DataFrame, params::Dict)
    local ra, dec
    if !("ra" in names(cat)) || !("dec" in names(cat))
        println("generating the coordinates of the sources")
        ra, dec = gen_radec(cat, params)
    else
        ra = cat.ra .* u"°"
        dec = cat.dec .* u"°"
    end
    
    ra_val = ustrip.(u"°", ra)
    dec_val = ustrip.(u"°", dec)

    pix_resol = params["pixel_size"] / 3600.

    ra_cen = 0.5 * (maximum(ra_val) + minimum(ra_val))
    dec_cen = 0.5 * (maximum(dec_val) + minimum(dec_val))
    delta_ra = maximum(ra_val) - minimum(ra_val)
    delta_dec = maximum(dec_val) - minimum(dec_val)

    w_celestial = WCSTransform(2)
    w_celestial.crval = [ra_cen, dec_cen]
    w_celestial.crpix = [0.5 * delta_ra / pix_resol, 0.5 * delta_dec / pix_resol]
    w_celestial.cdelt = [pix_resol, pix_resol]
    w_celestial.ctype = ["RA---TAN", "DEC--TAN"]
    w_celestial.cunit = ["deg", "deg"]

    worldcoords = vcat(ra_val', dec_val')
    pixcoords = world_to_pix(w_celestial, worldcoords)
    x = pixcoords[1, :]
    y = pixcoords[2, :]

    w_celestial.crpix = [
        0.5 * delta_ra / pix_resol - minimum(x),
        0.5 * delta_dec / pix_resol - minimum(y)
    ]

    pixcoords = world_to_pix(w_celestial, worldcoords)
    x = pixcoords[1, :]
    y = pixcoords[2, :]
    pos = [y, x]

    zmax = (params["freq_max"] - params["freq_min"]) / params["freq_resol"]

    shape_val = [Int(floor(zmax + 1)), Int(ceil(maximum(y))), Int(ceil(maximum(x)))]
    shape = [div(i, 2) * 2 + 1 for i in shape_val]

    x_edges = collect(-0.5:1.0:(shape[3] - 0.5))
    y_edges = collect(-0.5:1.0:(shape[2] - 0.5))
    z_edges = collect(-0.5:1.0:(shape[1] - 0.5))

    w = WCSTransform(3)
    w.crval = [ra_cen, dec_cen, params["freq_min"]]
    w.crpix = [w_celestial.crpix[1], w_celestial.crpix[2], 1.0]
    w.cdelt = [pix_resol, pix_resol, params["freq_resol"]]
    w.ctype = ["RA---TAN", "DEC--TAN", "FREQ"]
    w.cunit = ["deg", "deg", "Hz"]

    wcs_dict = Dict{String, Any}(
        "w" => w,
        "shape" => shape,
        "pos" => pos,
        "x_edges" => x_edges,
        "y_edges" => y_edges,
        "z_edges" => z_edges
    )
    return wcs_dict
end

function set_kernel(params::Dict, cube_prop_dict::Dict)
    w = cube_prop_dict["w"]
    N_chan = cube_prop_dict["shape"][1]
    kernel_size = cube_prop_dict["shape"][2]

    kernel = Matrix{Float64}[]
    beam_area_pix2 = Float64[]

    start_freq_idx = round(Int, w.crval[3] / w.cdelt[3])

    for chan_idx in 1:N_chan
        freq_idx = start_freq_idx + (chan_idx - 1)
        freq_Hz = freq_idx * w.cdelt[3]

        fwhm_rad = (1.22 * c_speed) / (freq_Hz * params["telescop_diameter"])
        fwhm_arcsec = fwhm_rad * (180.0 / pi) * 3600.0
        sigma_arcsec = fwhm_arcsec * gaussian_fwhm_to_sigma
        sigma_pix = sigma_arcsec / params["pixel_size"]

        N = kernel_size
        cx = (N + 1) / 2
        cy = (N + 1) / 2
        kernel_channel = zeros(Float64, N, N)
        
        for j in 1:N, i in 1:N
            dx = i - cx
            dy = j - cy
            kernel_channel[i, j] = exp(-(dx^2 + dy^2) / (2.0 * sigma_pix^2))
        end

        peak = maximum(kernel_channel)
        if peak > 0.0
            kernel_channel ./= peak
        end

        push!(kernel, kernel_channel)
        push!(beam_area_pix2, sum(kernel_channel))
    end
    return kernel, beam_area_pix2
end

# FITS export utilities for maps and cubes

function wcs_to_header_vectors(w::WCSTransform)
    header_str = WCS.to_header(w)
    h_keys = String[]
    h_vals = Any[]
    h_comms = String[]
    
    lines = split(header_str, '\n')
    for line in lines
        line = rstrip(line)
        if isempty(line) || line == "END"
            continue
        end
        if length(line) < 8
            continue
        end
        key = rstrip(line[1:8])
        if key == "CONTINUE" || key == "COMMENT" || key == "HISTORY"
            val_and_comment = length(line) > 8 ? line[9:end] : ""
            push!(h_keys, key)
            push!(h_vals, nothing)
            push!(h_comms, val_and_comment)
        else
            if length(line) >= 10 && line[9:10] == "= "
                value_part = line[11:end]
                parts = split(value_part, '/'; limit=2)
                raw_val = strip(parts[1])
                comm = length(parts) > 1 ? strip(parts[2]) : ""
                
                parsed_val = nothing
                if raw_val == "T"
                    parsed_val = true
                elseif raw_val == "F"
                    parsed_val = false
                elseif (startswith(raw_val, "'") && endswith(raw_val, "'")) || 
                       (startswith(raw_val, "\"") && endswith(raw_val, "\""))
                    parsed_val = raw_val[2:end-1]
                else
                    try
                        parsed_val = parse(Int, raw_val)
                    catch
                        try
                            parsed_val = parse(Float64, raw_val)
                        catch
                            parsed_val = raw_val
                        end
                    end
                end
                push!(h_keys, key)
                push!(h_vals, parsed_val)
                push!(h_comms, comm)
            else
                push!(h_keys, "COMMENT")
                push!(h_vals, nothing)
                push!(h_comms, line)
            end
        end
    end
    return h_keys, h_vals, h_comms
end

function save_map(filename::String, map_array::AbstractArray, map_prop_dict::Dict, filter_name::String, unit::String, beam_fwhm::Real, input_cat::String)
    println("Write $filename...")
    
    h_keys = String[]
    h_vals = Any[]
    h_comms = String[]
    
    function add_key!(k::String, v::Any, c::String="")
        push!(h_keys, k)
        push!(h_vals, v)
        push!(h_comms, c)
    end
    
    w = map_prop_dict["w"]
    add_key!("CRVAL1", w.crval[1])
    add_key!("CRVAL2", w.crval[2])
    add_key!("CRPIX1", w.crpix[1])
    add_key!("CRPIX2", w.crpix[2])
    add_key!("CDELT1", w.cdelt[1])
    add_key!("CDELT2", w.cdelt[2])
    add_key!("CTYPE1", w.ctype[1])
    add_key!("CTYPE2", w.ctype[2])
    add_key!("CUNIT1", w.cunit[1])
    add_key!("CUNIT2", w.cunit[2])
    
    add_key!("COMMENT", nothing, "map")
    add_key!("COMMENT", nothing, "Datas")
    add_key!("BUNIT", unit, "Physical unit of the map")
    add_key!("COMMENT", nothing, "Filter name = $filter_name")
    add_key!("COMMENT", nothing, "beam FWHM = $beam_fwhm arcsec")
    add_key!("COMMENT", nothing, "Input catalog = $input_cat")
    add_key!("DATE", string(Dates.now()), "Date of creation")
    
    header = FITSHeader(h_keys, h_vals, h_comms)
    
    FITS(filename, "w") do f
        write(f, map_array; header=header)
    end
end

function save_cubes(cube_input, cube_prop_dict::Dict, params_sides::Dict, params::Dict, component_name::String, just_save::Bool=false, just_compute::Bool=false)
    units_dict = Dict(
        "nobeam_Jy_pix" => "Jy/pixel",
        "nobeam_MJy_sr" => "MJy/sr",
        "smoothed_Jy_beam" => "Jy/beam",
        "smoothed_MJy_sr" => "MJy/sr"
    )
    
    cubes_dict = Dict{String, Any}()
    cubes2save = String[]
    
    if just_save == true
        cubes2save = collect(keys(cube_input))
        if get(params, "save_cube_nobeam_Jy_pix", false) == false
            filter!(e -> e != "nobeam_Jy_pix", cubes2save)
        end
        cubes_dict = cube_input
    else
        cubes_dict["nobeam_Jy_pix"] = cube_input
        if get(params, "save_cube_nobeam_Jy_pix", false) == true
            push!(cubes2save, "nobeam_Jy_pix")
        end
        
        if get(params, "gen_cube_nobeam_MJy_sr", false) == true
            pixel_sr = (params["pixel_size"] * pi / 180.0 / 3600.0)^2
            cubes_dict["nobeam_MJy_sr"] = (cube_input ./ pixel_sr) .* 1e-6
            push!(cubes2save, "nobeam_MJy_sr")
        end
        
        if get(params, "gen_cube_smoothed_Jy_beam", false) == true || 
           get(params, "gen_cube_smoothed_MJy_sr", false) == true
           
            println("Smooth the $component_name cube by the beam...")
            smoothed_Jybeam = deepcopy(cube_input)
            
            for f in 1:cube_prop_dict["shape"][1]
                slice = smoothed_Jybeam[f, :, :]
                kernel = cube_prop_dict["kernel"][f]
                smoothed_Jybeam[f, :, :] = real(ifft(fft(slice) .* fft(fftshift(kernel))))
            end
            
            cubes_dict["smoothed_Jy_beam"] = smoothed_Jybeam
            
            if get(params, "gen_cube_smoothed_Jy_beam", false) == true
                push!(cubes2save, "smoothed_Jy_beam")
            end
            
            if get(params, "gen_cube_smoothed_MJy_sr", false) == true
                smoothed_MJy_sr = zeros(Float64, size(smoothed_Jybeam)...)
                for f in 1:cube_prop_dict["shape"][1]
                    factor = cube_prop_dict["beam_area_pix2"][f] * (params["pixel_size"] * pi / 180.0 / 3600.0)^2 * 1e-6
                    smoothed_MJy_sr[f, :, :] = smoothed_Jybeam[f, :, :] ./ factor
                end
                cubes_dict["smoothed_MJy_sr"] = smoothed_MJy_sr
                push!(cubes2save, "smoothed_MJy_sr")
            end
        end
    end
    
    if !just_compute
        output_path = params["output_path"]
        
        for cube_type in cubes2save
            filename = joinpath(output_path, params["run_name"] * "_" * component_name * "_" * cube_type * ".fits")
            println("Write $filename...")
            
            if !isdir(output_path)
                println("Create $output_path")
                mkpath(output_path)
            end
            
            h_keys, h_vals, h_comms = wcs_to_header_vectors(cube_prop_dict["w"])
            
            function add_card!(k::String, v::Any, c::String="")
                push!(h_keys, k)
                push!(h_vals, v)
                push!(h_comms, c)
            end
            
            add_card!("COMMENT", nothing, "cube")
            add_card!("COMMENT", nothing, "Datas")
            add_card!("BUNIT", units_dict[cube_type], "Physical unit of the datacube")
            add_card!("COMMENT", nothing, "telescope diameter = $(params["telescop_diameter"])m")
            add_card!("COMMENT", nothing, "Input catalog = $(params["sides_cat_path"])")
            add_card!("DATE", string(Dates.now()), "date of the creation")
            
            header = FITSHeader(h_keys, h_vals, h_comms)
            
            FITS(filename, "w") do f
                write(f, cubes_dict[cube_type]; header=header)
            end
        end
    end
    return cubes_dict
end


# Main map and cube generating funtions


"""
    make_maps(cat::DataFrame, params_maps::Dict, params_sides::Dict)

Generates celestial maps for multiple filters, performing spatial binning 
and optional beam-smoothing convolution via FFT.
"""
function make_maps(cat::DataFrame, params_maps::Dict, params_sides::Dict)
    flux_filter_list = String[]
    output_path = params_maps["output_path"]
    
    if !isdir(output_path)
        println("Create $output_path")
        mkpath(output_path)
    end

    for (filter_name, pixel_size, beam_fwhm) in zip(params_maps["filter_list"], params_maps["pixel_size"], params_maps["beam_fwhm_list"])
        println("Generate the map for $filter_name...")

        Sname = "S" * filter_name
        push!(flux_filter_list, Sname)

        if !(Sname in names(cat))
            println("$filter_name fluxes are not included in the catalog. They are computed now...")
            params_temp = copy(params_sides)
            params_temp["filter_list"] = [filter_name]
            cat = gen_fluxes_filter(cat, params_temp)
        end

        println("Set World Coordinates System...")
        map_prop_dict = set_wcs_map(cat, pixel_size, params_sides)

        histo = compute_histogram2d(
            map_prop_dict["pos"][1],
            map_prop_dict["pos"][2],
            map_prop_dict["y_edges"],
            map_prop_dict["x_edges"],
            cat[!, Symbol(Sname)]
        )

        if get(params_maps, "gen_map_nobeam_Jy_pix", false) == true
            filename = joinpath(output_path, params_maps["run_name"] * "_" * filter_name * "_nobeam_Jy_pix.fits")
            save_map(filename, histo, map_prop_dict, filter_name, "Jy/pix", 0.0, params_maps["sides_cat_path"])
        end

        if get(params_maps, "gen_map_nobeam_MJy_sr", false) == true
            pixel_sr = (pixel_size * pi / 180.0 / 3600.0)^2
            map_temp = (histo ./ pixel_sr) .* 1e-6
            filename = joinpath(output_path, params_maps["run_name"] * "_" * filter_name * "_nobeam_MJy_sr.fits")
            save_map(filename, map_temp, map_prop_dict, filter_name, "MJy/sr", 0.0, params_maps["sides_cat_path"])
        end

        if get(params_maps, "gen_map_smoothed_Jy_beam", false) == true || 
           get(params_maps, "gen_map_smoothed_MJy_sr", false) == true

            println("Convolve the map by the beam...")
            sigma_pix = beam_fwhm * gaussian_fwhm_to_sigma / pixel_size

            ny = map_prop_dict["shape"][1]
            nx = map_prop_dict["shape"][2]
            cy = div(ny + 1, 2)
            cx = div(nx + 1, 2)
            kernel = zeros(Float64, ny, nx)
            
            for j in 1:nx, i in 1:ny
                dy = i - cy
                dx = j - cx
                kernel[i, j] = exp(-(dx^2 + dy^2) / (2.0 * sigma_pix^2))
            end

            histo_conv = real(ifft(fft(histo) .* fft(fftshift(kernel))))

            if get(params_maps, "gen_map_smoothed_Jy_beam", false) == true
                filename = joinpath(output_path, params_maps["run_name"] * "_" * filter_name * "_smoothed_Jy_beam.fits")
                save_map(filename, histo_conv, map_prop_dict, filter_name, "Jy/beam", beam_fwhm, params_maps["sides_cat_path"])
            end

            if get(params_maps, "gen_map_smoothed_MJy_sr", false) == true
                conv_factor = sum(kernel) * (pixel_size * pi / 180.0 / 3600.0)^2 * 1e-6
                map_temp = histo_conv ./ conv_factor
                filename = joinpath(output_path, params_maps["run_name"] * "_" * filter_name * "_smoothed_MJy_sr.fits")
                save_map(filename, map_temp, map_prop_dict, filter_name, "MJy/sr", beam_fwhm, params_maps["sides_cat_path"])
            end
        end
    end
    return cat
end

function channel_flux_densities(cat::DataFrame, params_sides::Dict, params::Dict)
    freq_min = params["freq_min"]
    freq_max = params["freq_max"]
    freq_resol = params["freq_resol"]
    
    num_steps = round(Int, 1.0 + (freq_max - freq_min) / freq_resol)
    channels = range(freq_min, stop=freq_max, length=num_steps)

    lambda_list = (c_speed * 1e6) ./ channels
    SED_dict = load_sed_pickle_equivalent(params_sides["SED_file"])

    println("Generate monochromatic fluxes...") #CONCERTO
    Snu_arr = gen_Snu_arr(
        lambda_list, 
        SED_dict, 
        cat.redshift, 
        cat.mu .* cat.LIR, 
        cat.Umean, 
        cat.Dlum, 
        cat.issb
    )
    return Snu_arr
end

function make_continuum_cube(cat::DataFrame, params_sides::Dict, params::Dict, cube_prop_dict::Dict)
    shape = cube_prop_dict["shape"]
    N_freq = shape[1]
    N_y = shape[2]
    N_x = shape[3]

    continuum_nobeam_Jypix = zeros(Float64, N_freq, N_y, N_x)
    channels_flux_densities = channel_flux_densities(cat, params_sides, params)
    
    pos_y = cube_prop_dict["pos"][1]
    pos_x = cube_prop_dict["pos"][2]
    y_edges = cube_prop_dict["y_edges"]
    x_edges = cube_prop_dict["x_edges"]

    for f in 1:N_freq
        row_weights = channels_flux_densities[:, f]
        histo = compute_histogram2d(pos_y, pos_x, y_edges, x_edges, row_weights)
        continuum_nobeam_Jypix[f, :, :] = histo
    end

    continuum_cubes = save_cubes(
        continuum_nobeam_Jypix, 
        cube_prop_dict, 
        params_sides, 
        params, 
        "continuum", 
        false, 
        !params["save_continuum_only"]
    )
    return continuum_cubes
end

function line_channel_flux_densities(line::String, rest_freq::Real, cat::DataFrame, cube_prop_dict::Dict)
    freq_obs = rest_freq ./ (1.0 .+ cat.redshift)

    w = cube_prop_dict["w"]
    N_gal = nrow(cat)
    
    worldcoords = Matrix{Float64}(undef, 3, N_gal)
    worldcoords[1, :] .= w.crval[1]
    worldcoords[2, :] .= w.crval[2]
    worldcoords[3, :] = freq_obs .* 1e9

    pixcoords = world_to_pix(w, worldcoords)
    channel = pixcoords[3, :]

    nudelt = abs(w.cdelt[3]) * 1e-9
    c_kms = c_speed * 1e-3
    vdelt = c_kms .* nudelt ./ freq_obs

    line_col = Symbol("I" * line)
    S = cat[!, line_col] ./ vdelt
    return S, channel
end

function make_co_cube(cat::DataFrame, params_sides::Dict, params::Dict, cube_prop_dict::Dict)
    first_Jup = 1
    last_Jup = 8
    
    local CO_all_cubes
    local keys_computed_cubes
    
    for J in first_Jup:last_Jup
        line_name = "CO$(J)$(J - 1)"
        println("Compute channel locations and flux densities of $line_name lines...")
        
        rest_freq = params_sides["nu_CO"] * J 
        Snu, channels = line_channel_flux_densities(line_name, rest_freq, cat, cube_prop_dict)

        println("Generate the non-smoothed $line_name cube...")
        CO_oneJ_nobeam_Jypix = compute_histogram3d(
            channels,
            cube_prop_dict["pos"][1],
            cube_prop_dict["pos"][2],
            cube_prop_dict["z_edges"],
            cube_prop_dict["y_edges"],
            cube_prop_dict["x_edges"],
            Snu
        )

        just_compute_oneJ = !get(params, "save_each_transition", false)
        CO_oneJ_cubes = save_cubes(
            CO_oneJ_nobeam_Jypix, 
            cube_prop_dict, 
            params_sides, 
            params, 
            line_name; 
            just_compute = just_compute_oneJ
        )

        if J == first_Jup
            keys_computed_cubes = collect(keys(CO_oneJ_cubes))
            CO_all_cubes = deepcopy(CO_oneJ_cubes)
        else
            for key in keys_computed_cubes
                CO_all_cubes[key] .+= CO_oneJ_cubes[key]
            end
        end
    end

    if get(params, "save_each_line", false) == true  
        println("Save the CO cubes containing all the transitions...")
        save_cubes(CO_all_cubes, cube_prop_dict, params_sides, params, "CO_all"; just_save = true)
    end
    return CO_all_cubes 
end

function make_cii_cube(cat::DataFrame, params_sides::Dict, params::Dict, cube_prop_dict::Dict, name_relation::String)
    println("Compute channel locations and flux densities of [CII] line ($name_relation et al. recipe)...")
    
    Snu, channels = line_channel_flux_densities("CII_" * name_relation, params_sides["nu_CII"], cat, cube_prop_dict)
    
    println("Generate the non-smoothed [CII] cube...")
    CII_nobeam_Jypix = compute_histogram3d(
        channels,
        cube_prop_dict["pos"][1],
        cube_prop_dict["pos"][2],
        cube_prop_dict["z_edges"],
        cube_prop_dict["y_edges"],
        cube_prop_dict["x_edges"],
        Snu
    )

    just_compute_cii = !get(params, "save_each_line", false)
    CII_cubes = save_cubes(
        CII_nobeam_Jypix, 
        cube_prop_dict, 
        params_sides, 
        params, 
        "CII_" * name_relation; 
        just_compute = just_compute_cii
    )
    return CII_cubes
end

function make_ci_cube(cat::DataFrame, params_sides::Dict, params::Dict, cube_prop_dict::Dict)
    line_names = ["CI10", "CI21"]
    first_loop = true
    
    local CI_both_cubes
    
    for line_name in line_names
        println("Compute channel locations and flux densities of [$line_name] lines...")
        
        rest_freq = params_sides["nu_" * line_name]
        Snu, channels = line_channel_flux_densities(line_name, rest_freq, cat, cube_prop_dict)
    
        println("Generate the non-smoothed [$line_name] cube...")
        CI_one_trans_nobeam_Jypix = compute_histogram3d(
            channels,
            cube_prop_dict["pos"][1],
            cube_prop_dict["pos"][2],
            cube_prop_dict["z_edges"],
            cube_prop_dict["y_edges"],
            cube_prop_dict["x_edges"],
            Snu
        )

        just_compute_ci = !get(params, "save_each_transition", false)
        CI_one_trans_cubes = save_cubes(
            CI_one_trans_nobeam_Jypix, 
            cube_prop_dict, 
            params_sides, 
            params, 
            line_name; 
            just_compute = just_compute_ci
        )

        if first_loop
            CI_both_cubes = deepcopy(CI_one_trans_cubes)
            first_loop = false
        else
            keys_computed_cubes = collect(keys(CI_both_cubes))
            for key in keys_computed_cubes
                CI_both_cubes[key] .+= CI_one_trans_cubes[key]
            end
        end
    end

    if get(params, "save_each_line", false) == true  
        println("Save the [CI] cubes containing all the transitions...")
        save_cubes(CI_both_cubes, cube_prop_dict, params_sides, params, "CI_both"; just_save = true)
    end
    return CI_both_cubes
end

"""
    make_cube(cat::DataFrame, params_sides::Dict, params_cube::Dict)

Constructs, smooths, and aggregates astronomical datacubes for continuum, CO, CI, and [CII] lines,
with optional combined line/continuum cubes according to specified parameters.
"""
function make_cube(cat::DataFrame, params_sides::Dict, params_cube::Dict)
    println("Set World Coordinates System...")
    cube_prop_dict = set_wcs(cat, params_cube)

    if get(params_cube, "gen_cube_smoothed_Jy_beam", false) == true || 
       get(params_cube, "gen_cube_smoothed_MJy_sr", false) == true
        println("Compute the beams for all channels...")
        kernel, beam_area_pix2 = set_kernel(params_cube, cube_prop_dict)
        cube_prop_dict["kernel"] = kernel
        cube_prop_dict["beam_area_pix2"] = beam_area_pix2
    end

    println("Create continuum cubes..")
    continuum_cubes = make_continuum_cube(cat, params_sides, params_cube, cube_prop_dict)

    println("Create CO cubes...")
    CO_cubes = make_co_cube(cat, params_sides, params_cube, cube_prop_dict)

    println("Create CI cubes...")
    CI_cubes = make_ci_cube(cat, params_sides, params_cube, cube_prop_dict)

    CII_relations_2compute = String[]
    keys_computed_cubes = collect(keys(continuum_cubes))

    if get(params_cube, "gen_cube_CII_Lagache", false) == true
        push!(CII_relations_2compute, "Lagache")
    end
    if get(params_cube, "gen_cube_CII_de_Looze", false) == true
        push!(CII_relations_2compute, "de_Looze")
    end

    for CII_relation_name in CII_relations_2compute
        CII_cubes = make_cii_cube(cat, params_sides, params_cube, cube_prop_dict, CII_relation_name)

        if get(params_cube, "save_all_lines", false) == true || get(params_cube, "save_full", false) == true
            combined_cubes = deepcopy(CO_cubes)

            println("Generate the cube(s) with all the lines...")
            for key in keys_computed_cubes
                combined_cubes[key] .+= CII_cubes[key]
                combined_cubes[key] .+= CI_cubes[key]
            end

            if get(params_cube, "save_all_lines", false) == true
                println("Save the cube(s) with all the lines...")
                save_cubes(combined_cubes, cube_prop_dict, params_sides, params_cube, "all_lines_$(CII_relation_name)", true)
            end

            if get(params_cube, "save_full", false) == true
                for key in keys_computed_cubes
                    combined_cubes[key] .+= continuum_cubes[key]
                end
                println("Save the full cube(s) containing continuum + lines...")
                save_cubes(combined_cubes, cube_prop_dict, params_sides, params_cube, "full_$(CII_relation_name)", true)
            end
        end
    end

    println("Done!")
    return true
end
