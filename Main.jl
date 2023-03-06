using ModelingToolkit, StochasticDiffEq
using Plots
# Define some variables
@parameters σ β
@variables t health(t) (pop(t))[1:2]
D = Differential(t)

eqs = [D(health) ~ 0,
       Symbolics.scalarize(D.(pop) .~ β .* health .* pop)...]
# eqs = [D(health) ~ 0]
noiseeqs = [σ,
            Symbolics.scalarize(0 .* pop)...]
# noiseeqs = [0.01σ]
@named de = SDESystem(eqs, noiseeqs, t, [health, pop...], [σ, β]; tspan=(0, 10.0))

# u0map = [health => 1.0,
#          pop => repeat([1.0], length(pop))]
u0map = [health => 1.0, Symbolics.scalarize(pop .=> 1.0)...]
parammap = [σ => 1.0,
            β => 0.2]

prob = SDEProblem(de, u0map, (0.0, 10.0), parammap)

sol = solve(prob, SOSRI())
plot(sol)
plot(sol; idxs=[health, pop])
