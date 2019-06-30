"""
Equilibrium Exact Diagonalization solver to compute Green's function
of the single orbital Anderson model on the Keldysh contour
"""

using Keldysh
using KeldyshED.Hilbert
using KeldyshED.Operators
using KeldyshED: EDCore, computegf
using ArgParse
using HDF5
using Distributed

function ArgParse.parse_item(::Type{Vector{T}}, str::AbstractString) where T
  map(s -> parse(T, s), split(str, ","))
end

function parse_commandline()
  s = ArgParseSettings(description =
    "ED solver for single orbital Anderson model with descrete bath")

  @add_arg_table s begin
    # Parameters of Keldysh grid
    "--tmax"
      help = "Maximal simulated time"
      arg_type = Float64
      default = 2.0
    "--beta"
      help = "Inverse temperature"
      arg_type = Float64
      default = 5.0
    "--npts_real"
      help = "Number of points on each real branch"
      arg_type = Int
      default = 41
    "--npts_imag"
      help = "Number of points on the imaginary branch"
      arg_type = Int
      default = 101
    # Model parameters
    "--e_d"
      help = "Local energy level"
      arg_type = Float64
      default = 1.0
    "--U"
      help = "Hubbard interaction constant"
      arg_type = Float64
      default = 2.0
    "--h"
      help = "Magnetic field"
      arg_type = Float64
      default = 0.0
    "--eps"
      help = "Positions of bath levels"
      arg_type = Vector{Float64}
      default = [-0.5, 0.5]
    "--V"
      help = "Hybridization constants"
      arg_type = Vector{Float64}
      default = [0.3, 0.3]
    # Output options
    "--gf.file"
      help = "Output HDF5 file with Green's function"
      arg_type = String
      default = "output.h5"
    "--gf.section"
      help = "Output HDF5 section with Green's function"
      arg_type = String
      default = "gf"
    "--params_section"
      help = "Output HDF5 section for used parameters"
      arg_type = String
      default = "parameters"
  end

  return parse_args(s)
end

function main(args)
  p = parse_commandline()

  println("Running with $(length(workers())) workers")

  β = p["beta"]
  ε = p["eps"]
  V = p["V"]

  @assert length(ε) == length(V)
  n_bath_sites = length(ε)

  spins = ("down", "up")
  soi = SetOfIndices([[s, a] for s in spins for a in 0:length(ε)])

  # Local Hamiltonian
  H_loc = p["e_d"] * (n("up", 0) + n("down", 0)) + p["U"] * n("up", 0) * n("down", 0)
  H_loc += -p["h"] * (n("up", 0) - n("down", 0))
  println("Local Hamiltonian: $H_loc")

  # Bath Hamiltonian
  H_bath = sum(e * (n("up", a) + n("down", a)) for (a, e) in enumerate(ε))
  println("Bath Hamiltonian: $H_bath")

  # Hybridization Hamiltonian
  H_hyb = OperatorExpr{Float64}()
  for s in spins
    H_hyb += sum(v * c_dag(s, a) * c(s, 0) for (a, v) in enumerate(V))
    H_hyb += sum(v * c_dag(s, 0) * c(s, a) for (a, v) in enumerate(V))
  end
  println("Hybridization Hamiltonian: $H_hyb")

  H = H_loc + H_bath + H_hyb

  println("Diagonalizing Hamiltonian...")
  ed, tottime, bytes, gctime, memallocs = @timed EDCore(H, soi)
  println("Diagonalized in $(tottime) sec (gctime = $(gctime) sec)")

  println("Found $(length(ed.subspaces)) invariant subspaces")
  println("Ground state energy: $(ed.gs_energy)")

  # Keldysh contour and grid
  contour = twist(Contour(full_contour, tmax = p["tmax"], β=β))
  grid = TimeGrid(contour, npts_real = p["npts_real"], npts_imag = p["npts_imag"])

  gf = Dict{String,TimeGF}()
  for s in spins
    println("Computing GF($s) ...")
    gf[s], tottime, bytes, gctime, memallocs = @timed computegf(ed,
                                                                grid,
                                                                IndicesType([s, 0]),
                                                                β)
    println("Computed in $(tottime) sec (gctime = $(gctime) sec)")
  end

  println("Saving results...")
  h5open(p["gf.file"], "w") do file
    # Write GF
    write(file, p["gf.section"] * "/0", gf["up"])
    write(file, p["gf.section"] * "/1", gf["down"])

    # Write ground state energy
    file["gs_energy"] = ed.gs_energy

    # Write parameters
    for (arg, val) in p
      file[p["params_section"] * "/$arg"] = val
    end
  end
end

main(ARGS)
