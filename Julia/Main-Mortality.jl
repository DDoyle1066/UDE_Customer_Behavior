using ModelingToolkit, StochasticDiffEq
using Plots
using Random
using SciMLBase
using StatsPlots
## Hypothetical mortatlity Example
#=
To-Do:
- Effect of wealth on constant and age increasing hazards
=#

@parameters μ_env μ_A μ_B Θ_region σ_region
pop_size = 10_000
num_regions = 4
@variables t
@variables ages_pop(t)[1:pop_size] cum_mort_pop(t)[1:pop_size]
# scalar that increases or decreases the 
@variables health_region(t)[1:num_regions,
                            1:num_regions]
@variables wealth_efficacy(t)
D = Differential(t)
rng = MersenneTwister(20230326)
loc_pop_x = Int.(ceil.(rand(rng, pop_size) .* num_regions))
loc_pop_y = Int.(ceil.(rand(rng, pop_size) .* num_regions))
health_region_pop = [health_region[x, y] for (x, y) in zip(loc_pop_x, loc_pop_y)]
# Equations for how age rolls forward
age_eqs = D.(ages_pop) .~ 1
mort_eqs = D.(cum_mort_pop) .~ (μ_A .* exp.(μ_B .* ages_pop) .+ μ_env) .* health_region_pop
age_noise_eqs = Symbolics.scalarize(0 .* ages_pop)
mort_noise_eqs = Symbolics.scalarize(0 .* cum_mort_pop)
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
       Symbolics.scalarize(region_eqs)...]
noiseeqs = [age_noise_eqs..., mort_noise_eqs..., region_noise_eqs...]
# noiseeqs = [0.01σ]
@named de = SDESystem(eqs, noiseeqs, t,
                      [ages_pop..., cum_mort_pop..., health_region...],
                      [μ_env, μ_A, μ_B, Θ_region, σ_region]; tspan=(0, 50.0))

# u0map = [health => 1.0,
#          pop => repeat([1.0], length(pop))]
init_ages = rand(rng, pop_size) .* 60
init_health = exp.(randn(rng, num_regions, num_regions) / 10)
death_chances = rand(rng, pop_size)
u0map = [Symbolics.scalarize(ages_pop .=> init_ages)...,
         Symbolics.scalarize(cum_mort_pop .=> 0)...,
         Symbolics.scalarize(health_region .=> init_health)...]
parammap = [μ_env => 7e-4,
            μ_A => 2e-5,
            μ_B => 0.1,
            Θ_region => 0.1,
            σ_region => 0.1]

prob = SDEProblem(de, u0map, (0.0, 50.0), parammap)
@time sol = solve(prob, SOSRI())
plot(sol; idxs=health_indices)
mean(sol(50; idxs=health_indices))
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
exp.(.-sol(50; idxs=mort_indices))

scatter(sol(50; idxs=ages_indices), exp.(.-sol(50; idxs=mort_indices)))