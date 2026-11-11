# Autoscalers.jl

Automated finite-size scaling analysis (data collapse) in Julia.

The [usage guide](generated/guide.md) collapses example data, estimates its parameters
with error bars and tests one of them against an expected value.

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
