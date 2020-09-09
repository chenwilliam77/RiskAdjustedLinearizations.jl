# Homotopy or Continuation algorithm
# Implement an SEIR criterion for choosing q
function solve_steadystate(m::AbstractDSGEModel, x0::AbstractVector{S1} = Vector{Float64}(undef, 0), q::Float64;
                           ftol::S2 = 1e-8, autodiff::Symbol = :autodiff, kwargs...) where {S1 <: Real, S2 <: Real}

    # Set up system of equations
    f3 = expectational_jump_coefs(m)
    f4 = expectational_state_coefs(m)
    h  = (z, y) -> expectational_nonlinearities(m, z, y)
    g  = (z, y) -> expected_state_transition(m, z, y)

    Ny, Nz = size(f4)

    N_zy   = Nz + Ny
    𝒱 = Vector{eltype(x0)}(undef, Nz)
    J𝒱 = Vector{eltype(x0)}(undef, Nz)
    𝒱_fnct! = (z, Ψ) -> entropy!(m, 𝒱, z, Ψ, f3, f4)
    J𝒱_fnct! = (z, Ψ) -> entropy_jacobian!(m, J𝒱, z, Ψ, f3, f4)

    _my_eqn = function _my_stochastic_equations(F, x)
        # Unpack
        z = @view x[1:Nz]
        y = @view x[(Nz + 1):N_zy]
        Ψ = reshape(@view x[(N_zy + 1):end], Nz, Nz)

        # Calculate entropy terms
        𝒱_fnct!(z,  Ψ)
        J𝒱_fnct!(z, Ψ)

        # Calculate Jacobian of nonlinear terms
        f1, f2 = expectational_jacobian(m, @view x[1:N_zy])
        g1, g2 = expected_state_transition_jacobian(m, @view x[1:N_zy])

        # Calculate residuals
        F[1:Nz] = g(z, y) - z
        F[(Nz + 1):N_zy] = h(z, y) + f3 * y + f4 * z + q * 𝒱
        F[(N_zy + 1):end] = f1 * Ψ + f2 + (f3 * Ψ + f4) * (g1 * Ψ + g2) + q * J𝒱
    end

    out = nlsolve(_my_eqn, x0, autodiff = autodiff, kwargs...)

    if out.f_converged
        return out.zero
    else
        error("A deterministic steady state could not be found.")
    end
end
