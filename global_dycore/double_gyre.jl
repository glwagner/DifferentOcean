using Oceananigans
using Oceananigans.Units
using Printf
using Enzyme

function time_step_double_gyre(model, surface_tracer_flux; Δt=10minutes)

    top_c_bc = model.tracers.c.boundary_conditions.top
    interior(top_c_bc.condition) .= surface_tracer_flux

    #=
    N² = 2e-5
    Δz = minimum_zspacing(grid)
    bᵢ(x, y, z) = N² * z + 1e-2 * N² * Δz * (2rand() - 1)
    uᵢ(x, y, z) = 1e-2 * (2rand() - 1)
    set!(model, b=bᵢ, u=uᵢ, v=vᵢ)
    =#

    # Initialize the model
    model.clock.iteration = 0
    model.clock.time = 0
    model.clock.last_Δt = Inf

    # This doesn't work and I'm not quite sure why:
    #
    # set!(model, c=0)
    # 
    # So we use fill! on parent(c) directly instead:
    fill!(parent(model.tracers.c), 0)

    # Step it forward
    for n = 1:10
        time_step!(model, Δt)
    end

    return model
end

arch = CPU()
Nx = 20
Ny = 20
Nz = 2

grid = LatitudeLongitudeGrid(arch,
                             size = (Nx, Ny, Nz),
                             halo = (7, 7, 7),
                             longitude = (-10, 10),
                             latitude = (30, 50),
                             z = [-2000, -400, 0])

momentum_advection = VectorInvariant()
surface_tracer_flux = Field{Center, Center, Nothing}(grid)
gaussian_flux(λ, φ) = - exp(-(λ^2 + (φ - 40)^2) / 18)
set!(surface_tracer_flux, gaussian_flux)
c_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(surface_tracer_flux))

# momentum_advection = WENOVectorInvariant(vorticity_scheme=WENO(order=5),
#                                          divergence_scheme=WENO(order=5),
#                                          vertical_scheme=Centered(order=2))
 
free_surface = ExplicitFreeSurface()

model = HydrostaticFreeSurfaceModel(; grid,
                                    coriolis = HydrostaticSphericalCoriolis(),
                                    momentum_advection,
                                    #tracer_advection = WENO(order=5),
                                    tracer_advection = Centered(order=2),
                                    closure = nothing,
                                    tracers = (:b, :c),
                                    boundary_conditions = (; c=c_bcs),
                                    buoyancy = BuoyancyTracer())

model = time_step_double_gyre(model, interior(surface_tracer_flux))

using GLMakie
heatmap(view(model.tracers.c, :, :, 2))
display(current_figure())

c_truth = Array(interior(model.tracers.c))

function estimate_tracer_error(model, surface_tracer_flux, c_truth)
    model = time_step_double_gyre(model, surface_tracer_flux)
    # Compute the sum square error
    c = model.tracers.c
    Nx, Ny, Nz = size(grid)
    err = 0.0
    for j = 1:Ny, i = 1:Nx
        err += @inbounds (c[i, j, Nz] - c_truth[i, j, Nz])^2
    end

    return err::Float64
end

# J is the unknown surface tracer flux
J = zeros(Nx, Ny)
e1 = estimate_tracer_error(model, J, c_truth)

i = 10
j = 10
ϵ = 1e-3
J[i, j] = ϵ
e2 = estimate_tracer_error(model, J, c_truth)
@show ΔeΔϵ = (e2 - e1) / ϵ
J[i, j] = 0

# Use autodiff to compute a gradient
dmodel = Enzyme.make_zero(model)
dJ = Enzyme.make_zero(J)
dedν = autodiff(set_runtime_activity(Enzyme.Reverse),
                estimate_tracer_error,
                Duplicated(model, dmodel),
                Duplicated(J, dJ),
                Const(c_truth))

