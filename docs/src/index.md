# Autoscalers.jl

Automated finite-size scaling analysis (data collapse) in Julia.

## Public interface

```@docs
@ansatz
Ansatz
Autoscaler
cost_breakdown
to_table
```

### Fitting and error estimates

```@docs
fit_collapse
bootstrap
BootstrapResult
confint(::BootstrapResult)
stderror(::BootstrapResult)
pvalue(::BootstrapResult, ::Pair{Symbol,<:Real}...)
```
