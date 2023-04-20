using Plots, Statistics
using Flux, Optimization, OptimizationFlux, DiffEqFlux, StochasticDiffEq,
      SciMLBase.EnsembleAnalysis

u0 = Float32[2.0; 0.0]
datasize = 30
tspan = (0.0f0, 1.0f0)
tsteps = range(tspan[1], tspan[2]; length=datasize)

function trueSDEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    return du .= ((u .^ 3)'true_A)'
end

mp = Float32[0.2, 0.2]
function true_noise_func(du, u, p, t)
    return du .= mp .* u
end

prob_truesde = SDEProblem(trueSDEfunc, true_noise_func, u0, tspan)

# Take a typical sample from the mean
ensemble_prob = EnsembleProblem(prob_truesde)
ensemble_sol = solve(ensemble_prob, SOSRI(); trajectories=10000)
ensemble_sum = EnsembleSummary(ensemble_sol)

sde_data, sde_data_vars = Array.(timeseries_point_meanvar(ensemble_sol, tsteps))

drift_dudt = Flux.Chain(x -> x .^ 3,
                        Flux.Dense(2, 50, tanh),
                        Flux.Dense(50, 2))
p1, re1 = Flux.destructure(drift_dudt)

diffusion_dudt = Flux.Chain(Flux.Dense(2, 2))
p2, re2 = Flux.destructure(diffusion_dudt)

neuralsde = NeuralDSDE(drift_dudt, diffusion_dudt, tspan, SOSRI();
                       saveat=tsteps, reltol=1e-1, abstol=1e-1)

# Get the prediction using the correct initial condition
prediction0 = neuralsde(u0)

drift_(u, p, t) = re1(p[1:(neuralsde.len)])(u)
diffusion_(u, p, t) = re2(p[(neuralsde.len + 1):end])(u)

prob_neuralsde = SDEProblem(drift_, diffusion_, u0, (0.0f0, 1.2f0), neuralsde.p)

ensemble_nprob = EnsembleProblem(prob_neuralsde)
ensemble_nsol = solve(ensemble_nprob, SOSRI(); trajectories=100,
                      saveat=tsteps)
ensemble_nsum = EnsembleSummary(ensemble_nsol)

plt1 = plot(ensemble_nsum; title="Neural SDE: Before Training")
scatter!(plt1, tsteps, sde_data'; lw=3)

scatter(tsteps, sde_data[1, :]; label="data")
scatter!(tsteps, prediction0[1, :]; label="prediction")

function predict_neuralsde(p, u=u0)
    return Array(neuralsde(u, p))
end

function loss_neuralsde(p; n=100)
    u = repeat(reshape(u0, :, 1), 1, n)
    samples = predict_neuralsde(p, u)
    means = mean(samples; dims=2)
    vars = var(samples; dims=2, mean=means)[:, 1, :]
    means = means[:, 1, :]
    loss = sum(abs2, sde_data - means) + sum(abs2, sde_data_vars - vars)
    return loss, means, vars
end

list_plots = []
iter = 0

# Callback function to observe training
callback = function (p, loss, means, vars; doplot=false)
    global list_plots, iter

    if iter == 0
        list_plots = []
    end
    iter += 1

    # loss against current data
    display(loss)

    # plot current prediction against data
    plt = Plots.scatter(tsteps, sde_data[1, :]; yerror=sde_data_vars[1, :],
                        ylim=(-4.0, 8.0), label="data")
    Plots.scatter!(plt, tsteps, means[1, :]; ribbon=vars[1, :], label="prediction")
    push!(list_plots, plt)

    if doplot
        display(plt)
    end
    return false
end

opt = ADAM(0.025)

# First round of training with n = 10
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_neuralsde(x; n=10), adtype)
optprob = Optimization.OptimizationProblem(optf, neuralsde.p)
result1 = Optimization.solve(optprob, opt;
                             callback=callback, maxiters=100)