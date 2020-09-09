using SafeTestsets

# Start Test Script

@time begin
    @time @safetestset "Utilies" begin
        include("util.jl")
    end

    @time @safetestset "Risk-Adjusted Linearization" begin
        include("risk_adjusted_linearization/constructors.jl")
    end

    @time @safetestset "Solution algorithms" begin
        include("solution_algorithms/blanchard_kahn.jl")
        include("solution_algorithms/compute_psi.jl")
    end
end
