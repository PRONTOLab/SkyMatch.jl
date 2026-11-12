# ==============================================================================
# SIDES Pipeline: Complete Abundance Matching Script
# ==============================================================================
# This script performs abundance matching between the stellar mass functions (SMFs)
# and halo mass functions (HMFs) using the UCHUU ultra-deep lightcone.
# ==============================================================================

using HDF5
using DataFrames
using Cosmology
using DelimitedFiles
using Interpolations
using Roots
using PythonCall
using Plots
using Dates
using Logging

# Disable verbose warnings if necessary
disable_logging(Logging.Warn)

# --- 1. Python Interoperability Setup via PythonCall ---
# We load numpy and the AbundanceMatching library from the Python environment
const np = pyimport("numpy")
const abundancematching = pyimport("AbundanceMatching")

# --- 2. Local Script Includes & Parameters ---
# Include necessary helper files for coordinate and parameter loading
include("/home/agkogkou/docs/scripts/z_from_Rcom.jl")
include("/home/agkogkou/sides/PYSIDES/pysides/load_params.jl")

# --- 3. Cosmology & Constants Setup ---
const DEFAULT_COSMOLOGY = "Planck15"
# Setup the flat LCDM cosmology matching Planck18 parameters
const cosmo = Cosmology.flatLCDM(h=0.6774, OmegaM=0.3075)
const h_hub = cosmo.h

# --- 4. Load & Process the UCHUU Halo Catalog ---
println("Loading the UCHUU lightcone...")

# Load the HDF5 datasets directly into a Julia DataFrame
df = h5open("/data/SIDES/UCHUU/lightcone1_mpeak.hdf5", "r") do file
    m200c = read(file, "M200C")
    rcom = read(file, "Rcom")
    
    # Check for optional columns, fallback to default vectors if missing
    ra = haskey(file, "RA") ? read(file, "RA") : zeros(Float64, length(rcom))
    dec = haskey(file, "DEC") ? read(file, "DEC") : zeros(Float64, length(rcom))
    z = haskey(file, "z") ? read(file, "z") : zeros(Float64, length(rcom))
    id_vals = haskey(file, "ID") ? read(file, "ID") : collect(1:length(rcom))
    mpeak = haskey(file, "Mpeak") ? read(file, "Mpeak") : copy(m200c)
    
    DataFrame(M200C=m200c, Rcom=rcom, RA=ra, DEC=dec, z=z, ID=id_vals, Mpeak=mpeak)
end

println("Loaded DataFrame with $(nrow(df)) halos.")

# Filter halos: Cut at 1e10 M_sun / h (where UCHUU begins missing halos)
filter!(row -> (row.M200C / h_hub) > 1e10, df)
println("DataFrame size after mass cut: $(nrow(df)) halos.")

# Sort catalog by comoving distance (Rcom)
println("Sorting with respect to column Rcom...")
sort!(df, :Rcom)
println("Sorted.")

# Rescale comoving distance and mass columns in-place by the Hubble parameter
df.Rcom ./= h_hub
df.M200C ./= h_hub

# --- 5. Create the Redshift Snapshot Grid ---
# Comoving snapshot distances corresponding to each UCHUU snapshot
const Rbins_com = Float64[
    0, 150, 300, 450, 600, 750, 900, 1050, 1200, 1350, 1500, 1650, 1800, 1950, 2100, 
    2250, 2400, 2550, 2700, 2850, 3000, 3150, 3300, 3450, 3600, 3750, 3900, 4050, 
    4200, 4350, 4500, 4650, 4800, 4950, 5100, 5250, 5400, 5550, 5700, 5850, 5972
]

Rbins = Rbins_com ./ h_hub

"""
Helper function to compute redshift (z) at a target comoving distance (Mpc).
Solves: comoving_radial_dist(z) - r_target = 0 using bisection.
"""
function z_at_value_comoving(c, r_target::Real)
    r_target <= 0.0 && return 0.0
    # Find root within range z in [0.0, 20.0]
    f_root(z) = Cosmology.comoving_radial_dist(c, z) - r_target
    return find_zero(f_root, (0.0, 20.0))
end

# Compute the redshift bounds for each bin
zbins = [z_at_value_comoving(cosmo, r) for r in Rbins]
zbins[1] = 0.0 # Force first bin edge exactly to 0.0

# Calculate grid points at bin midpoints
zgrid = zbins[1:end-1] .+ diff(zbins) ./ 2.0
zmin, zmax = minimum(zgrid), maximum(zgrid)

# --- 6. Load SMF Parameters & Build SMF Grids ---
println("Loading the SMFs...")
params = load_params("/home/agkogkou/sides/PYSIDES/PAR_FILES/SIDES_unclustered_1deg2_standard.par")
params["path_smf_evo_file"] = "/home/agkogkou/sides/PYSIDES/pysides/SMF_SIDES_2017.txt"

# Read double-Schechter parameters from evolution file
smf_data = readdlm(params["path_smf_evo_file"])
zmean = smf_data[:, 1]
logMknee = smf_data[:, 2]
phiknee1 = smf_data[:, 3]
phiknee2 = smf_data[:, 4]
alpha1 = smf_data[:, 5]
alpha2 = smf_data[:, 6]

params["logMcut"] = 5.0
Nz = length(zgrid)

# Calculate cosmological luminosity distances and volumes
Dlum_bins = [Cosmology.luminosity_dist(cosmo, z) for z in zbins]
const uchuu_fsize = 9.03 * 13.60 # ultra-deep field size in deg^2
const omega = (pi / 180.0)^2 * uchuu_fsize
Vbins = (omega / 3.0) .* diff((Dlum_bins ./ (1.0 .+ zbins)) .^ 3)

# Define target stellar mass grid
Npts_mass = Int((params["logMmax"] - params["logMcut"]) * params["Npts_Mstar_dex"])
Mstargrid = 10.0 .^ (params["logMcut"] .+ (params["logMmax"] - params["logMcut"]) / Npts_mass .* collect(0:Npts_mass))

# Find index range of active redshifts
inzrange = findall(z -> zmin <= z <= zmax, zgrid)

# Helper function to create Interpolation objects
function make_interp(x_ref, y_ref)
    return LinearInterpolation(x_ref, y_ref, extrapolation_bc=Line())
end

# Interpolate Schechter parameters onto our target redshift grid midpoint values
Mknee_z = 10.0 .^ make_interp(1.0 .+ zmean, logMknee).(1.0 .+ zgrid[inzrange])
phiknee1_z = make_interp(1.0 .+ zmean, phiknee1).(1.0 .+ zgrid[inzrange])
alpha1_z = make_interp(1.0 .+ zmean, alpha1).(1.0 .+ zgrid[inzrange])
alpha2_z = make_interp(1.0 .+ zmean, alpha2).(1.0 .+ zgrid[inzrange])
phiknee2_z = exp.(make_interp(log.(1.0 .+ zmean), log.(phiknee2)).(log.(1.0 .+ zgrid[inzrange])))

# Cache the Stellar Mass Functions (SMFs) and cumulative SMFs
SMF = Matrix{Float64}[]
SMFcum = Matrix{Float64}[]
Phis = Vector{Float64}[]

for k in 1:length(zgrid)
    # Double-Schechter formula
    ratio = Mstargrid ./ Mknee_z[k]
    term1 = phiknee1_z[k] .* (ratio .^ alpha1_z[k])
    term2 = phiknee2_z[k] .* (ratio .^ alpha2_z[k])
    
    phi = exp.(-ratio) .* (term1 .+ term2) ./ Mknee_z[k] .* Mstargrid .* log(10.0)
    push!(Phis, phi)
    
    # 2D Tabular SMF
    smf = hcat(Mstargrid, phi)
    push!(SMF, smf)
    
    # Backward cumulative sum for cumulative SMF
    diff_log_M = diff(log10.(Mstargrid))
    weighted_phi = diff_log_M .* phi[1:end-1]
    ccum = reverse(cumsum(reverse(weighted_phi)))
    
    push!(SMFcum, hcat(Mstargrid[1:end-1], ccum))
end

println("All SMFs loaded and successfully cached!")

# --- 7. Abundance Matching Loop ---
println("Abundance Matching started...")

# Preallocate a vector to collect calculated stellar masses
Mst_sc = Float64[]

for i in 1:(length(Rbins) - 1)
    z1 = zbins[i]
    z2 = zbins[i+1]
    println("[$(i)/$(length(Rbins)-1)] - AM for $(round(z1, digits=2)) < z < $(round(z2, digits=2)) in progress...")

    # 1. Sanitize the current slice's SMF
    raw_smfy = SMF[i][:, 2] ./ 2.5
    raw_smfx = -2.5 .* log10.(SMF[i][:, 1]) # Convert masses to fake magnitudes

    # Drop NaNs, Infs, and zeroes in a single optimized pass
    valid_indices = findall(idx -> begin
        y = raw_smfy[idx]
        x = raw_smfx[idx]
        !isnan(y) && !isinf(y) && y != 0.0 && !isnan(x) && !isinf(x)
    end, eachindex(raw_smfy))

    smfy = raw_smfy[valid_indices]
    smfx = raw_smfx[valid_indices]

    # Mirroring PySIDES: Zip and flip the arrays
    lf = hcat(reverse(smfx), reverse(smfy))
    Mmin = minimum(smfx)
    Mmax = maximum(smfx)

    # Initialize the AbundanceFunction via PythonCall
    af = abundancematching.AbundanceFunction(Py(lf[:, 1]), Py(lf[:, 2]), (Mmin, Mmax), nbin=10000)

    # 2. Deconvolution (scatter separation)
    scatter_val = 0.2
    LF_SCATTER_MULT = 2.5
    r_repeats = 50
    remainder = af.deconvolute(scatter_val * LF_SCATTER_MULT, r_repeats)

    # 3. Filter the UCHUU catalog for the comoving bin range
    bdf = filter(row -> Rbins[i] <= row.Rcom < Rbins[i+1], df)

    # Calculate bin volume
    Dl1 = Cosmology.luminosity_dist(cosmo, z1)
    Dl2 = Cosmology.luminosity_dist(cosmo, z2)
    Vz = (omega / 3.0) * ((Dl2 / (1.0 + z2))^3 - (Dl1 / (1.0 + z1))^3)
    box_size = Vz^(1.0 / 3.0)

    # 4. Perform the matching
    nd_halos_v = abundancematching.calc_number_densities(Py(bdf.Mpeak), box_size)
    Mag_sc = af.match(nd_halos_v, scatter_val * LF_SCATTER_MULT)

    # Convert results back to native Julia Vector
    Mag_sc_jl = pyconvert(Vector{Float64}, Mag_sc)
    Mst_count = 10.0 .^ (-Mag_sc_jl ./ 2.5)

    # Safety validation check
    count_test = count(isnan, Mst_count)
    if count_test > 0
        println("ERROR!!! There are $count_test nan values for $(round(z1, digits=2)) < z < $(round(z2, digits=2))")
        exit(1)
    else
        println("NO nan values - we are GOOD!")
    end

    # Append chunk stellar masses in-place to avoid GC reallocations
    append!(Mst_sc, Mst_count)
end

# --- 8. Export Catalog and Update global DataFrame ---
println("Saving the ID and Mst_sc values in a 2d array...")
idm = hcat(df.ID, Mst_sc)

out_dir = "/home/agkogkou/docs/AMcatalogs"
if !isdir(out_dir)
    mkpath(out_dir)
end

# Save to disk using Python NumPy directly via PythonCall
np.save(joinpath(out_dir, "Mst_Mpeak_comp.npy"), Py(idm))
println("Job FINISHED!")

# Add the final column to the global DataFrame
println("All the stellar masses were created and assigned to the halos")
df[!, :Mstar] = Mst_sc
println("New column (:Mstar) added to the total data frame!")
