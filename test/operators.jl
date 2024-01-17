# KeldyshED.jl
#
# Copyright (C) 2019 Igor Krivenko <igor.s.krivenko@gmail.com>
# Copyright (C) 2015 P. Seth, I. Krivenko, M. Ferrero and O. Parcollet
#
# KeldyshED.jl is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# KeldyshED.jl is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# KeldyshED.jl. If not, see <http://www.gnu.org/licenses/>.
#
# Author: Igor Krivenko

using Test

using KeldyshED.Operators

function test_comm_anticomm(Cd::Vector{O}, C::Vector{O}) where O
  S = scalartype(O)
  ref_anticomm(i, j) = "$(i == j ? one(S) : zero(S))"
  ref_comm(i, j) = (i == j ? "$(-one(S)) + " : "") * "$(2*one(S))*c†($i)c($j)"

  for (i, cd) in enumerate(Cd)
    for (j, c) in enumerate(C)
      @test repr(cd*c + c*cd) == ref_anticomm(i, j)
      @test repr(cd*c - c*cd) == ref_comm(i, j)
    end
  end
end

@testset "Real OperatorExpr" begin
  # Operators without indices
  op_with_no_indices = c() + c_dag() - n()
  @test repr(op_with_no_indices) == "1.0*c†() + 1.0*c() + -1.0*c†()c()"

  # Operators with many indices
  op_with_many_indices = c(1, 2, "a", "x", -2) + c_dag(3, 15, "b", "y", -5)
  @test repr(op_with_many_indices) == "1.0*c†(3,15,\"b\",\"y\",-5)" *
                                      " + 1.0*c(1,2,\"a\",\"x\",-2)"

  # Constant operator
  const_op = RealOperatorExpr(3.14)
  @test repr(const_op) == "3.14"

  # Test anticommutators & commutators
  test_comm_anticomm([c_dag(i) for i in 1:4], [c(i) for i in 1:4])

  # Algebra
  x = c(0)
  y = c_dag(1)
  @test repr(x) == "1.0*c(0)"
  @test repr(y) == "1.0*c†(1)"
  @test repr(-x)  == "-1.0*c(0)"
  @test repr(x + 2.0) == "2.0 + 1.0*c(0)"
  @test repr(2.0 + x) == "2.0 + 1.0*c(0)"
  @test repr(x - 2.0) == "-2.0 + 1.0*c(0)"
  @test repr(2.0 - x) == "2.0 + -1.0*c(0)"
  @test repr(3.0 * y) == "3.0*c†(1)"
  @test repr(y * 3.0) == "3.0*c†(1)"
  @test repr(x + y) == "1.0*c†(1) + 1.0*c(0)"
  @test repr(x - y) == "-1.0*c†(1) + 1.0*c(0)"
  @test repr((x + y) * (x - y)) == "2.0*c†(1)c(0)"

  # Nested algebra
  @test 0.25*n(1) == 0.5*(0.5*n(1))

  # N^3
  N  = n("up") + n("dn");
  N3 = N * N * N;
  @test repr(N) == "1.0*c†(\"dn\")c(\"dn\") + 1.0*c†(\"up\")c(\"up\")"
  @test repr(N3) == "1.0*c†(\"dn\")c(\"dn\") + 1.0*c†(\"up\")c(\"up\")" *
                    " + 6.0*c†(\"dn\")c†(\"up\")c(\"up\")c(\"dn\")"

  # Dagger
  X = c_dag(1) * c_dag(2) * c(3) * c(4)
  @test repr(X) == "-1.0*c†(1)c†(2)c(4)c(3)"
  @test repr(dagger(X)) == "-1.0*c†(3)c†(4)c(2)c(1)"


  # map()
  @test repr(map((m,c) -> 2c, X)) == "-2.0*c†(1)c†(2)c(4)c(3)"
  @test repr(map((m,c) -> 0c, X)) == "0.0"

  # Real & imaginary parts
  @test repr(real(X)) == "-1.0*c†(1)c†(2)c(4)c(3)"
  @test repr(imag(X)) == "0.0"
end

@testset "Complex OperatorExpr" begin
  Cmplx = Complex{Float64}

  # Operators without indices
  op_with_no_indices = c(scalar_type = Cmplx) +
                       c_dag(scalar_type = Cmplx) -
                       n(scalar_type = Cmplx)
  @test repr(op_with_no_indices) == "(1.0 + 0.0im)*c†() + (1.0 + 0.0im)*c()" *
                                    " + (-1.0 - 0.0im)*c†()c()"

  # Operators with many indices
  op_with_many_indices = c(1, 2, "a", "x", -2, scalar_type = Cmplx) +
                         c_dag(3, 15, "b", "y", -5, scalar_type = Cmplx)
  @test repr(op_with_many_indices) == "(1.0 + 0.0im)*c†(3,15,\"b\",\"y\",-5)" *
                                      " + (1.0 + 0.0im)*c(1,2,\"a\",\"x\",-2)"

  # Constant operator
  const_op = ComplexOperatorExpr(3.14 + 2im)
  @test repr(const_op) == "(3.14 + 2.0im)"

  # Test anticommutators & commutators
  test_comm_anticomm([c_dag(i) for i in 1:4], [c(i) for i in 1:4])

  # Algebra
  x = c(0, scalar_type = Cmplx)
  y = c_dag(1, scalar_type = Cmplx)
  @test repr(x) == "(1.0 + 0.0im)*c(0)"
  @test repr(y) == "(1.0 + 0.0im)*c†(1)"
  @test repr(-x)  == "(-1.0 - 0.0im)*c(0)"
  @test repr(x + 2.0) == "(2.0 + 0.0im) + (1.0 + 0.0im)*c(0)"
  @test repr(2.0 + x) == "(2.0 + 0.0im) + (1.0 + 0.0im)*c(0)"
  @test repr(x + 2.0im) == "(0.0 + 2.0im) + (1.0 + 0.0im)*c(0)"
  @test repr(2.0im + x) == "(0.0 + 2.0im) + (1.0 + 0.0im)*c(0)"
  @test repr(x - 2.0) == "(-2.0 + 0.0im) + (1.0 + 0.0im)*c(0)"
  @test repr(2.0 - x) == "(2.0 + 0.0im) + (-1.0 - 0.0im)*c(0)"
  @test repr(x - 2.0im) == "(-0.0 - 2.0im) + (1.0 + 0.0im)*c(0)"
  @test repr(2.0im - x) == "(0.0 + 2.0im) + (-1.0 - 0.0im)*c(0)"
  @test repr(3.0 * y) == "(3.0 + 0.0im)*c†(1)"
  @test repr(y * 3.0) == "(3.0 + 0.0im)*c†(1)"
  @test repr(3.0im * y) == "(0.0 + 3.0im)*c†(1)"
  @test repr(y * 3.0im) == "(0.0 + 3.0im)*c†(1)"
  @test repr(x + y) == "(1.0 + 0.0im)*c†(1) + (1.0 + 0.0im)*c(0)"
  @test repr(x - y) == "(-1.0 - 0.0im)*c†(1) + (1.0 + 0.0im)*c(0)"
  @test repr((x + y) * (x - y)) == "(2.0 + 0.0im)*c†(1)c(0)"

  # N^3
  N  = n("up", scalar_type = Cmplx) + n("dn", scalar_type = Cmplx);
  N3 = N * N * N;
  @test repr(N) == "(1.0 + 0.0im)*c†(\"dn\")c(\"dn\")" *
                   " + (1.0 + 0.0im)*c†(\"up\")c(\"up\")"
  @test repr(N3) == "(1.0 + 0.0im)*c†(\"dn\")c(\"dn\")" *
                    " + (1.0 + 0.0im)*c†(\"up\")c(\"up\")" *
                    " + (6.0 + 0.0im)*c†(\"dn\")c†(\"up\")c(\"up\")c(\"dn\")"

  # Dagger
  X = (1.0 + 2.0im) * c_dag(1, scalar_type = Cmplx) *
                      c_dag(2, scalar_type = Cmplx) *
                      c(3, scalar_type = Cmplx) *
                      c(4, scalar_type = Cmplx)
  @test repr(X) == "(-1.0 - 2.0im)*c†(1)c†(2)c(4)c(3)"
  @test repr(dagger(X)) == "(-1.0 + 2.0im)*c†(3)c†(4)c(2)c(1)"

  # map()
  @test repr(map((m,c) -> 2c, X)) == "(-2.0 - 4.0im)*c†(1)c†(2)c(4)c(3)"
  @test repr(map((m,c) -> 0c, X)) == "0.0 + 0.0im"

  # Real & imaginary parts
  @test repr(real(X)) == "(-1.0 + 0.0im)*c†(1)c†(2)c(4)c(3)"
  @test repr(imag(X)) == "(-2.0 + 0.0im)*c†(1)c†(2)c(4)c(3)"
end
