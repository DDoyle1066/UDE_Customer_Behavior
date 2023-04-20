using Plots
using Random
using SciMLBase
using StatsPlots
using Statistics
using DataDrivenDiffEq
using OrdinaryDiffEq
using DataDrivenSR
using DiffEqFlux
using Lux
using Flux
include("src/Mortality.jl")
using ..Mort: Mort
## Hypothetical mortatlity Example
#=
To-Do:
- Fit parameters with traditional methods
- Fit parameters with deep learning methods
- Fit parameters with diffEq methods
- Regime Switching
- Update parameter fitting
=#
tspan = (0.0, 100.0)
ps = Mort.gen_params(; μ_env=7e-4, μ_A=2e-5, μᵦ₁=0.02, μᵦ₂=0.03, μᵦ₃=0.05,
                     d1_cure_chance=0.05, d2_cure_chance=0.05, d3_cure_chance=0.05,
                     cure_1_eff=0.2, cure_2_eff=0.1, cure_3_eff=0.05,
                     pop_size=10_000)
u0 = Mort.generate_u0(ps)
ode_prob = ODEProblem(Mort.true_drift!, u0, tspan, ps)
@time sol = solve(ode_prob, Tsit5(); saveat=0:100);
sol_dead = Lux.gpu.(Mort.gen_mort_data(sol))
neural_prob = Mort.gen_model(10, u0, tspan; device=Lux.gpu)
@time neural_sol_arr = Mort.neural_sol(neural_prob, sol)
@time Mort.loss(neural_prob, sol_dead)
