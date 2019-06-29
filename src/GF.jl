using KeldyshED: EDCore
using Keldysh
using LinearAlgebra: Diagonal, tr

export computegf

"""Compute Green's function on a given Keldysh contour at inverse temperature β"""
function computegf(ed::EDCore, grid::TimeGrid, indices::IndicesType, β)
  gf = TimeGF(grid)

  en = energies(ed)
  ρ = density_matrix(ed, β)

  for t1 in grid, t2 in grid
    greater = θ(t1.val, t2.val)
    Δt = greater ? (t1.val.val - t2.val.val) : (t2.val.val - t1.val.val)

    conn_f = greater ? cdag_connection : c_connection
    left_mat_f, right_mat_f = greater ? (c_matrix, cdag_matrix) :
                                        (cdag_matrix, c_matrix)

    val::ComplexF64 = 0
    for outer_sp = 1:length(ed.subspaces)
      inner_sp = conn_f(ed, indices, outer_sp)
      isnothing(inner_sp) && continue

      # Eigenvalues in outer and inner subspaces
      outer_energies = en[outer_sp]
      inner_energies = en[inner_sp]

      right_mat = right_mat_f(ed, indices, outer_sp)
      outer_exp = Diagonal(exp.(1im * outer_energies * Δt))
      right_mat = right_mat * outer_exp

      left_mat = left_mat_f(ed, indices, inner_sp)
      inner_exp = Diagonal(exp.(-1im * inner_energies * Δt))
      left_mat = left_mat * inner_exp

      val += tr(ρ[outer_sp] * left_mat * right_mat)
    end
    gf[t1, t2] = (greater ? -1im : 1im) * val
  end
  gf
end
