module Mort
using Random
using SparseArrays
using StatsBase
using OrdinaryDiffEq
using Lux
using ComponentArrays
# defined indices for parameters
μ_env_ind, μ_A_ind, μᵦ₁_ind, μᵦ₂_ind, μᵦ₃_ind,
cure_1_eff_ind, cure_2_eff_ind, cure_3_eff_ind = 1:8
function gen_indices(pop_size::Int)
    age_indices = 1:pop_size
    mort_indices = ((maximum(age_indices) + 1):(maximum(age_indices) + pop_size))
    wealth_indices = ((maximum(mort_indices) + 1):(maximum(mort_indices) + pop_size))
    global ind = (; age=age_indices, mort=mort_indices, wealth=wealth_indices)
    return nothing
end
function thiele(ages;
                a=0.02474,
                b=0.3,
                c=0.002,
                d=0.5,
                e=25,
                f=2e-5,
                g=0.1)
    return @. a * exp(-b * ages) + c * exp(-0.5 * d * (ages - e)^2) + f * exp(g * ages)
end
function gen_params(; μ_env, μ_A, μᵦ₁, μᵦ₂, μᵦ₃,
                    d1_cure_chance, d2_cure_chance, d3_cure_chance,
                    cure_1_eff, cure_2_eff, cure_3_eff,
                    pop_size::Int, seed=1)
    rng = MersenneTwister(seed)
    global d1_cure_time = -log(rand(rng)) / d1_cure_chance
    global d2_cure_time = -log(rand(rng)) / d2_cure_chance
    global d3_cure_time = -log(rand(rng)) / d3_cure_chance
    Mort.gen_indices(pop_size)
    global pop_μᵦ = zeros(pop_size)
    ps = zeros(cure_3_eff_ind)
    ps = [μ_env, μ_A, μᵦ₁, μᵦ₂, μᵦ₃,
          cure_1_eff, cure_2_eff, cure_3_eff]

    return ps
end
function generate_u0(ps; seed=1)
    rng = MersenneTwister(seed)
    u0 = zeros(maximum(ind.wealth))
    pop_size = length(pop_μᵦ)
    u0[ind.age] .= rand(rng, pop_size) * 60
    u0[ind.mort] .= 0
    u0[ind.wealth] .= sample(rng, 1:20, pop_size)
    return u0
end
function true_drift!(du, u, p, t)
    μ_env = p[μ_env_ind]
    μ_A, μᵦ₁, μᵦ₂, μᵦ₃ = p[μ_A_ind], p[μᵦ₁_ind], p[μᵦ₂_ind], p[μᵦ₃_ind]
    pop_μᵦ .= 0
    ## population age and mortality
    du[ind.age] .= 1
    ## determining population exposure  
    if t >= d1_cure_time
        if t >= d1_cure_time + 20
            wealth_pct = max(min((t - d1_cure_time - 20), 6), 0) + 10
            pop_μᵦ .+= μᵦ₁ .*
                       ifelse.(u[ind.wealth] .<= wealth_pct, ((1 - p[cure_1_eff_ind])),
                               1.0)
        else
            pop_μᵦ .+= μᵦ₁ .* ifelse.(u[ind.wealth] .<= 4, (1 - p[cure_1_eff_ind]), 1.0)
        end
    else
        pop_μᵦ .+= μᵦ₁
    end
    if t >= d2_cure_time
        if t >= d2_cure_time + 20
            wealth_pct = max(min((t - d2_cure_time - 20), 6), 0) + 10
            pop_μᵦ .+= μᵦ₂ .*
                       ifelse.(u[ind.wealth] .<= wealth_pct, (1 - p[cure_1_eff_ind]),
                               1.0)
        else
            pop_μᵦ .+= μᵦ₂ .* ifelse.(u[ind.wealth] .<= 4, (1 - p[cure_1_eff_ind]), 1.0)
        end
    else
        pop_μᵦ .+= μᵦ₂
    end
    if t >= d3_cure_time
        if t >= d3_cure_time + 20
            wealth_pct = max(min((t - d3_cure_time - 20), 6), 0) + 10
            pop_μᵦ .+= μᵦ₃ .*
                       ifelse.(u[ind.wealth] .<= wealth_pct, ((1 - p[cure_1_eff_ind])), 1.0)
        else
            pop_μᵦ .+= μᵦ₃ .* ifelse.(u[ind.wealth] .<= 4, (1 - p[cure_1_eff_ind]), 1.0)
        end
    else
        pop_μᵦ .+= μᵦ₃
    end
    du[ind.mort] .= (μ_A .* exp.(pop_μᵦ .* u[ind.age]) .+ μ_env)
    return nothing
end
# sol = Main.sol
function gen_mort_data(sol::ODESolution; seed=1)
    rng = MersenneTwister(seed)
    cum_mort_probs = 1 .- exp.(.-hcat([x[ind.mort] for x in sol.u]...))
    death_chance = rand(rng, size(cum_mort_probs)[1])
    dead = cum_mort_probs .> death_chance
    died = (dead[:, 2:end] .== 1) .& (dead[:, 1:(end - 1)] .== 0)
    @assert sum(died) == sum(dead[:, end])
    return dead[:, 1:(end - 1)], died
end
function transform_x(x)
    u, t = x
    # println(size(vcat(x .^3,t)))
    return vcat(u[ind.age]', u[ind.wealth]', repeat([t], length(ind.wealth))')
end
function gen_model(hidden_size, u0, tspan; seed=1, device=cpu)
    @assert device ∈ [cpu, gpu] "Device should be one of gpu or cpu"
    rng = MersenneTwister(seed)
    model = Lux.Chain(transform_x,
                      Lux.Dense(3, hidden_size, Lux.relu),
                      Lux.Dense(hidden_size, 1, exp))
    p, st = Lux.setup(rng, model)
    p = device(ComponentArray(p))
    st = device(st)
    function dudt!(du, u, p, t)
        du[ind.age] .= 1
        du[ind.mort] .= vec(model((u, t), p, st)[1]')
        return du
    end
    prob = ODEProblem(dudt!, device(u0), tspan, p)
    return prob, p, model
end
neural_sol(neural_prob, sol) = Array(solve(neural_prob, Tsit5(); saveat=sol.t))
#=
neural_prob = Main.neural_prob
sol_dead = Main.sol_dead
=#
function loss(neural_prob, sol, sol_dead)
    neural_sol_arr = neural_sol(neural_prob, sol)
    neural_mort_probs = min.(1 - 1.0f-6,
                             max.(1.0f-6,
                                  1 .-
                                  exp.(.-(neural_sol_arr[ind.mort, 2:end] .-
                                          neural_sol_arr[ind.mort, 1:(end - 1)]))))
    log_loss = -sum((log.(neural_mort_probs) .* (sol_dead[2]) .+ log.(1 .- neural_mort_probs) .* (1 .- sol_dead[2]))[sol_dead[1] .== 0])
    return log_loss
end
end