# Universal Differential Equations to Forecast Customer Behavior
This project uses a [Universal Differential Equation (UDE) framework](./refs/Universal_Differential_Equations_for_Scientific_Machine_Learning.pdf) to forecast customer behavior that follows a hypothetical framework. Specifically, this project examines the relative efficacy of traditional, deep learning and UDE methods in forecasting consumer behavior using data from the [M5 Forecasting competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) where the goal was to provide confidence intervals around consumer demand predictions.


# Installation
This project uses Julia 1.8.5. [juliaup](https://github.com/JuliaLang/juliaup) is the recommeneded manager for Julia installations.
To install:
- Clone the repository with `git clone https://github.com/DDoyle1066/UDE_Customer_Behavior`
- Activate the environment and install packages with:
```julia-repl
julia> ]

(@v1.8) pkg> activate .

(UDE_Customer_Behavior) pkg> instantiate
```

