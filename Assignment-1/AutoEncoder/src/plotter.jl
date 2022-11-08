using Plots
Base.@kwdef mutable struct Lorenz
    dt::Float64 = 0.02
    σ::Float64 = 10
    ρ::Float64 = 28
    β::Float64 = 8 / 3
    x::Float64 = 1
    y::Float64 = 1
    z::Float64 = 1
end

function step(l::Lorenz)
    dx = l.σ * (l.y - l.x)
    dy = l.x * (l.ρ - l.z) - l.y
    dz = l.x * l.y - l.β * l.z
    l.x += l.dt * dx
    l.y += l.dt * dy
    l.z += l.dt * dz
end

attractor = Lorenz()

plt = plot3d(
    1,
    xlims=(-30, 30),
    ylims=(-30, 30),
    zlim=(0, 60),
    title="Lorenz Attractor",
    marker=2
)
@gif for i = 1:1500
    step(attractor)
    push!(plt, attractor.x, attractor.y, attractor.z)
end every 100