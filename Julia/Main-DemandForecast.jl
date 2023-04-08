using CSV: CSV
using DataFrames
using CategoricalArrays
using DataConvenience
## M5-Forecasting Example
sales_data = CSV.read("data/sales_train_evaluation.csv", DataFrame)
sell_prices = CSV.read("data/sell_prices.csv", DataFrame)
calendar = CSV.read("data/calendar.csv", DataFrame)
transform!(sales_data, :dept_id => CategoricalArray => :dept_id)
transform!(sales_data, :cat_id => CategoricalArray => :cat_id)
transform!(sales_data, :store_id => CategoricalArray => :store_id)
transform!(sales_data, :state_id => CategoricalArray => :state_id)
transform!(sales_data, :item_id => CategoricalArray => :item_id)
transform!(sales_data, :dept_id => (x -> levelcode.(x)) => :dept_id_num)
transform!(sales_data, :cat_id => (x -> levelcode.(x)) => :cat_id_num)
transform!(sales_data, :store_id => (x -> levelcode.(x)) => :store_id_num)
transform!(sales_data, :state_id => (x -> levelcode.(x)) => :state_id_num)
transform!(sales_data, :item_id => (x -> levelcode.(x)) => :item_id_num)
function process_sales(sales, day_start, day_end)
    included_cols = [:id, :item_id, :dept_id, :cat_id, :store_id, :state_id]
    day_cols = Symbol.("d_" .* string.(day_start:day_end))
    long_sales = stack(sales[:, vcat(included_cols..., day_cols...)],
                       day_cols)
    leftjoin!(long_sales,
              calendar[:, Not([:date, :weekday])];
              on=[:variable => :d])
    leftjoin!(long_sales, sell_prices; on=[:wm_yr_wk, :store_id, :item_id])

    return long_sales
end
processed_sales = process_sales(sales_data, 1, 50)

#=
To-Do:
 - Create JLD2 objects that can be interacted with by the model 
 - Build elasticcity model for a single item in a single store
 - Expand to multiple items in the same dept in a single store
 - Exapnd to all items in a single store
Move across stores
=#