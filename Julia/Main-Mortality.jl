using Plots
using Random
using SciMLBase
using StatsPlots
using Statistics
using DataDrivenDiffEq
using OrdinaryDiffEq
using DataDrivenSR
using DiffEqFlux
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
                     pop_size=100)
u0 = Mort.generate_u0(ps)
ode_prob = ODEProblem(Mort.true_drift!, u0, tspan, ps)
@time sol = solve(ode_prob, Tsit5(); saveat=0:100);
sol_dead = Flux.cpu.(Mort.gen_mort_data(sol))

model = Chain(Mort.transform_x,
              Dense(3, 50, Flux.relu),
              Dense(50, 1, exp))
ps = Flux.params(model)
neural_prob = Mort.gen_model(10, u0, tspan; device=cpu)
@time neural_sol_arr = Mort.neural_sol(neural_prob, sol)
@time Mort.loss(neural_prob, sol, sol_dead)
ps = Params(neural_prob.p)
g = gradient(() -> Mort.loss(neural_prob, sol, sol_dead), ps)
g = gradient(Mort.loss, neural_prob, sol, sol_dead)

gs = back((one(l), nothing))[1]
rng = MersenneTwister(1234)
opt = tstate = Lux.Training.TrainState(rng, model, opt; transform_variables=Lux.gpu)
vjp_rule = Lux.Training.ZygoteVJP()
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> Mort.loss(x, sol, sol_dead), adtype)
optprob = Optimization.OptimizationProblem(optf, p)
result_neuralode = Optimization.solve(optprob, Adam(0.05); maxiters=300)

using Lux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL, Random,
      Plots
using ComponentArrays
rng = MersenneTwister(1)
ages = 0:100
exposures = repeat100_000 .* exp.(.-cumsum(vcat(0, Mort.thiele(ages[1:(end - 1)], 0))))
u0 = Matrix(hcat(exposures, ages)')
datasize = 31
tspan = (0.0f0, 30.0f0)
tsteps = range(tspan[1], tspan[2]; length=datasize)

function trueODEfunc(du, u, p, t)
    du[2, :] .= 0
    du[1, :] .= .-(1 .- exp.(.-Mort.thiele(u[2, :], t))) .* u[1, :]
    return nothing
end

prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob_trueode, Tsit5(); saveat=tsteps))
ode_data_prob = (1 .- ode_data[1, :, 2:end] ./ ode_data[1, :, 1:(end - 1)])
weights = ode_data[1, :, 1:(end - 1)]

p_x, st_x = Lux.setup(rng, x)
x(u0, p_x, st_x)[1]
nn_μ = Lux.Chain(Lux.Dense(2, 10, tanh),
                 Lux.Dense(10, 1, exp))
dd_test = Lux.Chain(x -> nn_μ(x))
p_test, st_test = Lux.setup(rng, dd_test)
dd_test(u0, p_test, st_test)
dudt2 = Lux.SkipConnection(Lux.Chain(Lux.Dense(2, 10, tanh),
                                     Lux.Dense(10, 1, exp)),
                           (x, y) -> cat((-x .* y[1]), ones(size(x)); dims=1))

p, st = Lux.setup(rng, dudt2)
dudt2(u0, p, st)[1]
prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(); saveat=tsteps)

function predict_neuralode(p)
    return Array(prob_neuralode(u0, p, st)[1])
end
predict_neuralode(p)
function loss_neuralode(p)
    pred = predict_neuralode(p)
    true_mort = 1 .- ode_data[1, :, 2:end] ./ ode_data[1, :, 1:(end - 1)]
    pred_mort = min.(max.(1 .- pred[1, :, 2:end] ./ pred[1, :, 1:(end - 1)], 1.0f-6),
                     1 - 1.0f-6)
    log_loss = @. (log(pred_mort) * true_mort + log(1 - pred_mort) * (1 - true_mort)) *
                  weights
    mean_log_loss = -mean(log_loss)
    return mean_log_loss, pred
end

# Do not plot by default for the documentation
# Users should change doplot=true to see the plots callbacks
callback = function (p, l, pred; doplot=false)
    println(l)
    # plot current prediction against data
    if doplot
        plt = scatter(tsteps, ode_data[1, :]; label="data")
        scatter!(plt, tsteps, pred[1, :]; label="prediction")
        display(plot(plt))
    end
    return false
end

pinit = ComponentArray(p)
callback(pinit, loss_neuralode(pinit)...; doplot=false)

# use Optimization.jl to solve the problem
adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

result_neuralode = Optimization.solve(optprob,
                                      ADAM(0.5);
                                      callback=callback,
                                      maxiters=300)

optprob2 = remake(optprob; u0=result_neuralode.u)
loss_neuralode(result_neuralode.u)
result_neuralode2 = Optimization.solve(optprob2,
                                       Optim.BFGS(; initial_stepnorm=0.01);
                                       callback=callback,
                                       allow_f_increases=false)
ode_data[1, :, end]
dim_stds = std(ode_data; dims=[2, 3])
loss = sum(abs2, (ode_data .- pred) ./ dim_stds; dims=[2, 3])
var(ode_data; dims=[2, 3])
pred = predict_neuralode(result_neuralode.u)
ode_data[1, :, end]
pred[1, :, end]
callback(result_neuralode.u, loss_neuralode(result_neuralode.u)...; doplot=true)
callback(result_neuralode2.u, loss_neuralode(result_neuralode2.u)...; doplot=true)