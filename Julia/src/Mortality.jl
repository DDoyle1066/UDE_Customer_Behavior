module Mort
using Random
using SparseArrays
using StatsBase
function get_indices(pop_size::Int, num_regions::Int; seed=1)
    rng = MersenneTwister(seed)
    age_indices = 1:pop_size
    mort_indices = ((maximum(age_indices) + 1):(maximum(age_indices) + pop_size))
    wealth_indices = ((maximum(mort_indices) + 1):(maximum(mort_indices) + pop_size))
    health_indices = zeros(Int, num_regions, num_regions)
    for i in 1:num_regions
        for j in 1:num_regions
            health_indices[i, j] = i + (j - 1) * num_regions + maximum(wealth_indices)
        end
    end
    wealth_eff_index = maximum(health_indices) + 1
    loc = sample(rng, health_indices, pop_size)
    return (; age=age_indices, mort=mort_indices, wealth=wealth_indices,
            health=health_indices, wealth_eff=wealth_eff_index, loc=loc)
end
function gen_params(; μ_env, μ_A, μ_B,
                    Θ_region, σ_region,
                    μ_wealth, σ_wealth, Θ_wealth, Θ_wealth_eff, σ_wealth_eff,
                    pop_size::Int, num_regions::Int)
    ps = (; μ_env, μ_A, μ_B,
          Θ_region, σ_region,
          μ_wealth, σ_wealth, Θ_wealth,
          Θ_wealth_eff, σ_wealth_eff,
          ind=Mort.get_indices(pop_size, num_regions),
          pop_size, num_regions)
    return ps
end
function true_drift!(du, u, p, t)
    μ_A, μ_B, μ_env = p.μ_A, p.μ_B, p.μ_env
    Θ_region = p.Θ_region
    μ_wealth, Θ_wealth = p.μ_wealth, p.Θ_wealth
    Θ_wealth_eff = p.Θ_wealth_eff
    num_regions = p.num_regions
    ## Propogate health from region to region
    for i in 1:num_regions
        for j in 1:num_regions
            avg_surrounging_health = 0
            num_neighbors = 0
            if i > 1
                avg_surrounging_health += u[p.ind.health[i - 1, j]]
                num_neighbors += 1
            end
            if i < num_regions
                avg_surrounging_health += u[p.ind.health[i + 1, j]]
                num_neighbors += 1
            end
            if j > 1
                avg_surrounging_health += u[p.ind.health[i, j - 1]]
                num_neighbors += 1
            end
            if j < num_regions
                avg_surrounging_health += u[p.ind.health[i, j + 1]]
                num_neighbors += 1
            end
            du[p.ind.health[i, j]] = Θ_region * (avg_surrounging_health / num_neighbors -
                                                 u[p.ind.health[i, j]])
        end
    end
    ## wealth efficacy
    du[p.ind.wealth_eff] = Θ_wealth_eff * (1 - u[p.ind.wealth_eff])
    ## population wealth
    du[p.ind.wealth] .= μ_wealth .* u[p.ind.wealth] .+ Θ_wealth .* (1 .- u[p.ind.wealth])
    ## population age and mortality
    du[p.ind.age] .= 1
    du[p.ind.mort] .= (μ_A .* exp.(μ_B .* u[p.ind.age]) .+ μ_env) ./# Gompertz-Makeham mortality
                      u[p.ind.loc] # higher health regions die slower

    return nothing
end
function true_noise!(du, u, p, t)
    du[p.ind.health] .= u[p.ind.health] .* p.σ_region
    du[p.ind.wealth] .= u[p.ind.wealth] .* p.σ_wealth
    du[p.ind.wealth_eff] = u[p.ind.wealth_eff] * p.σ_wealth_eff
    return nothing
end
function generate_u0(ps; seed=1)
    rng = MersenneTwister(seed)
    u0 = zeros(ps.ind.wealth_eff)
    u0[ps.ind.age] .= rand(rng, ps.pop_size) * 60
    u0[ps.ind.mort] .= 0
    u0[ps.ind.wealth] .= exp.(randn(rng, ps.pop_size) / 10)
    u0[ps.ind.health] .= exp.(randn(rng, size(ps.ind.health)) ./ 10)
    u0[ps.ind.wealth_eff] = 1.0
    return u0
end

end