using ModelingToolkit, StochasticDiffEq
using Plots
using Random
using SciMLBase
using StatsPlots
## Hypothetical mortatlity Example
#=
To-Do:
- Effect of wealth on constant and age increasing hazards
- Regime Switching
=#

@parameters μ_env μ_A μ_B
@parameters Θ_region σ_region
@parameters μ_wealth σ_wealth Θ_wealth_eff σ_wealth_eff
all_params = (; μ_env, μ_A, μ_B, Θ_region, σ_region, μ_wealth, σ_wealth, Θ_wealth_eff,
              σ_wealth_eff)
pop_size = 1_000
num_regions = 4
@variables t
@variables ages_pop(t)[1:pop_size] cum_mort_pop(t)[1:pop_size] wealth_pop(t)[1:pop_size]
# scalar that increases or decreases the 
@variables health_region(t)[1:num_regions,
                            1:num_regions]
@variables wealth_efficacy(t)
all_vars = (; ages_pop, cum_mort_pop, wealth_pop, health_region, wealth_efficacy)
D = Differential(t)
rng = MersenneTwister(20230326)
loc_pop_x = Int.(ceil.(rand(rng, pop_size) .* num_regions))
loc_pop_y = Int.(ceil.(rand(rng, pop_size) .* num_regions))
health_region_pop = [health_region[x, y] for (x, y) in zip(loc_pop_x, loc_pop_y)]
# Equations for how age rolls forward
age_eqs = D.(ages_pop) .~ 1
mort_eqs = D.(cum_mort_pop) .~ (μ_A .* exp.(μ_B .* ages_pop) .+ μ_env) .*
                               health_region_pop .* wealth_efficacy .*
                               exp.((wealth_pop .- sum(wealth_pop) / length(wealth_pop)) ./
                                    sqrt(sum(wealth_pop)^2 / length(wealth_pop) -
                                         sum(wealth_pop) / length(wealth_pop)))
wealth_eqs = D.(wealth_pop) .~ μ_wealth
wealth_eff_eq = D(wealth_efficacy) ~ Θ_wealth_eff * (1 - wealth_efficacy)
age_noise_eqs = Symbolics.scalarize(0 .* ages_pop)
mort_noise_eqs = Symbolics.scalarize(0 .* cum_mort_pop)
wealth_noise_eqs = Symbolics.scalarize(σ_wealth .* wealth_pop)
wealth_eff_noise_eq = Symbolics.scalarize(σ_wealth_eff * wealth_efficacy)
# Equations for how health propagates from region to region
region_eqs = Matrix{Equation}(undef, num_regions, num_regions)
for i in 1:num_regions
    for j in 1:num_regions
        avg_surrounging_health = 0
        num_neighbors = 0
        if i > 1
            avg_surrounging_health += health_region[i - 1, j]
            num_neighbors += 1
        end
        if i < num_regions
            avg_surrounging_health += health_region[i + 1, j]
            num_neighbors += 1
        end
        if j > 1
            avg_surrounging_health += health_region[i, j - 1]
            num_neighbors += 1
        end
        if j < num_regions
            avg_surrounging_health += health_region[i, j + 1]
            num_neighbors += 1
        end
        region_eqs[i, j] = D(health_region[i, j]) ~ Θ_region * (avg_surrounging_health /
                                                                num_neighbors -
                                                                health_region[i, j])
    end
end
region_noise_eqs = Symbolics.scalarize(health_region .* σ_region)
eqs = [Symbolics.scalarize(age_eqs)...,
       Symbolics.scalarize(mort_eqs)...,
       Symbolics.scalarize(wealth_eqs)...,
       Symbolics.scalarize(region_eqs)...,
       wealth_eff_eq]
noiseeqs = [age_noise_eqs..., mort_noise_eqs..., wealth_noise_eqs..., region_noise_eqs...,
            wealth_eff_noise_eq]
# noiseeqs = [0.01σ]
@named de = SDESystem(eqs, noiseeqs, t,
                      vcat([vcat(x...) for x in all_vars]...),
                      [x for x in all_params]; tspan=(0, 100.0))

# u0map = [health => 1.0,
#          pop => repeat([1.0], length(pop))]
init_ages = rand(rng, pop_size) .* 60
init_health = exp.(randn(rng, num_regions, num_regions) / 10)
init_wealth = randn(rng, pop_size)
death_chances = rand(rng, pop_size)
u0map = [Symbolics.scalarize(all_vars.ages_pop .=> init_ages)...,
         Symbolics.scalarize(all_vars.cum_mort_pop .=> 0)...,
         Symbolics.scalarize(all_vars.wealth_pop .=> init_wealth)...,
         Symbolics.scalarize(all_vars.health_region .=> init_health)...,
         all_vars.wealth_efficacy => 1.0]
parammap = [μ_env => 7e-4,
            μ_A => 2e-5,
            μ_B => 0.1,
            Θ_region => 0.1,
            σ_region => 0.1,
            μ_wealth => 0,
            σ_wealth => 0.2,
            Θ_wealth_eff => 0.1,
            σ_wealth_eff => 0.2]

prob = SDEProblem(de, u0map, (0.0, 100.0), parammap)
@time sol = solve(prob, SOSRI())

ages_indices = zeros(Int, length(ages_pop))
syms = SciMLBase.getsyms(sol)
for i in 1:length(ages_pop)
    ages_indices[i] = SciMLBase.sym_to_index(ages_pop[i], syms)
end
mort_indices = zeros(Int, length(cum_mort_pop))
for i in 1:length(cum_mort_pop)
    mort_indices[i] = SciMLBase.sym_to_index(cum_mort_pop[i], syms)
end
health_indices = zeros(Int, size(health_region))
for i in 1:num_regions
    for j in 1:num_regions
        health_indices[i, j] = SciMLBase.sym_to_index(health_region[i, j], syms)
    end
end
num_dead_over_time = Vector{Int}(undef, length(sol.t))
for (i, t) in enumerate(sol.t)
    num_dead_over_time[i] = sum(death_chances .> exp.(.-sol(t; idxs=mort_indices)))
end
plot(sol.t, num_dead_over_time)
exp.(.-sol(100; idxs=mort_indices))
plot(sol; idxs=health_indices)
mean(sol(50; idxs=health_indices))

scatter(sol(100; idxs=ages_indices), exp.(.-sol(100; idxs=mort_indices)))