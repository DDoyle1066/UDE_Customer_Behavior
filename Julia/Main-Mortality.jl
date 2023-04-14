using StochasticDiffEq
using Plots
using Random
using SciMLBase
using StatsPlots
using Statistics
using DataDrivenDiffEq
using OrdinaryDiffEq
using DataDrivenSR
include("src/Mortality.jl")
using ..Mort: Mort
## Hypothetical mortatlity Example
#=
To-Do:
- Fit parameters with traditional methods
- Fit parameters with diffEq methods
- Regime Switching
- Update parameter fitting
=#

ps = Mort.gen_params(; μ_env=7e-4, μ_A=2e-5, μ_B=0.1,
                     Θ_region=0.1, σ_region=0.1,
                     μ_wealth=0, σ_wealth=0.2, Θ_wealth=0.2,
                     Θ_wealth_eff=0.1, σ_wealth_eff=0.2,
                     pop_size=1_000, num_regions=10)
u0 = Mort.generate_u0(ps)
sde_prob = SDEProblem(Mort.true_drift!, Mort.true_noise!, u0, (0.0, 100.0), ps)
@time sol = solve(sde_prob, SOSRI(); saveat=0:100);
plot(sol; idxs=ps.ind.wealth_eff)
sol(100; idxs=ps.ind.wealth)
