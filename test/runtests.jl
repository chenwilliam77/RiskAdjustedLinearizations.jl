using SafeTestsets

# Start Test Script

@time begin
#=    @time @safetestset "Utilities" begin
        include("util.jl")
    end

    @time @safetestset "Risk-Adjusted Linearization" begin
        include("risk_adjusted_linearization/constructors.jl")
    end

    @time @safetestset "Numerical algorithms" begin
        include("numerical_algorithms/blanchard_kahn.jl")
        include("numerical_algorithms/compute_psi.jl")
        include("numerical_algorithms/deterministic.jl")
        include("numerical_algorithms/relaxation.jl")
        include("numerical_algorithms/homotopy.jl")
        include("numerical_algorithms/solve.jl")
    end=#
end
