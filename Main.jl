using ModelingToolkit, StochasticDiffEq
using Plots
using CSV: CSV
using DataFrames
sales_data = CSV.read("data/sales_train_evaluation.csv", DataFrame)
sell_prices = CSV.read("data/sell_prices.csv", DataFrame)
calendar = CSV.read("data/calendar.csv", DataFrame)

if !isfile("data/long_sales.csv")
    long_sales = stack(sales_data,
                       Symbol.(names(sales_data)[7:end]))
    leftjoin!(long_sales,
              calendar[:, Not([:date, :weekday])];
              on=[:variable => :d])
    leftjoin!(long_sales, sell_prices; on=[:wm_yr_wk, :store_id, :item_id])
    CSV.write("data/long_sales.csv", long_sales)
else
    long_sales = CSV.read("data/long_sales.csv", DataFrame)
end
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
