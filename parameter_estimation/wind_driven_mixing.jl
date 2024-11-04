using Oceananigans
using Oceananigans.Units
using Oceananigans.Simulations: reset!
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities:
    CATKEVerticalDiffusivity,
    CATKEMixingLength,
    TKEDissipationVerticalDiffusivity

using ParameterEstimocean.Parameters: closure_with_parameters
using Enzyme

grid = RectilinearGrid(size=50, z=(-200, 0), topology=(Flat, Flat, Bounded))

coriolis = FPlane(f=1e-4)
u_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(1e-3))
closure = CATKEVerticalDiffusivity()
tracers = (:b, :e)
buoyancy = BuoyancyTracer()
model = HydrostaticFreeSurfaceModel(; grid, closure, coriolis,
                                    tracers, buoyancy,
                                    boundary_conditions=(; u=u_bcs))

N² = 1e-5
bᵢ(z) = N² * z
set!(model, b=bᵢ)

simulation = Simulation(model, Δt=1minute, stop_time=12hours)
run!(simulation)

using GLMakie
lines(model.tracers.b)

b_truth = Array(interior(model.tracers.b))

function wind_driven_error(simulation, Cˢ, b_truth) 

    model = simulation.model
    mixing_length = CATKEMixingLength(; Cˢ)
    new_closure = CATKEVerticalDiffusivity(; mixing_length)
    model.closure = new_closure

    # Initialize model
    reset!(simulation)
    simulation.stop_time = 12hours
    set!(model, b=bᵢ, u=0, v=0, e=0)
    run!(simulation)

    b = model.tracers.b
    Nx, Ny, Nz = size(model.grid)
    err = 0.0
    for k = 1:Nz
        err += @inbounds (b[1, 1, k] - b_truth[1, 1, k])^2
    end

    return err::Float64
end


ϵ = 1e-1
Cˢ1 = 2.0
Cˢ2 = Cˢ1 + ϵ
e1 = wind_driven_error(simulation, Cˢ1, b_truth)
lines!(model.tracers.b)
e2 = wind_driven_error(simulation, Cˢ2, b_truth)
lines!(model.tracers.b)

@show ΔeΔC = (e2 - e1) / ϵ

# Use autodiff to compute a gradient
dsim = Enzyme.make_zero(simulation)
dedC = autodiff(set_runtime_activity(Enzyme.Reverse),
                wind_driven_error,
                Duplicated(simulation, dsim),
                Active(Cˢ1),
                Const(b_truth))

