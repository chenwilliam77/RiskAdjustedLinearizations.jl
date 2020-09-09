# RiskAdjustedLinearizations

[![GitHub release](https://img.shields.io/github/release/chenwilliam77/RiskAdjustedLinearizations.jl.svg)](https://github.com/chenwilliam77/RiskAdjustedLinearizations.jl/releases/latest)
[![](https://img.shields.io/badge/docs-stable-3f51b5.svg)](https://chenwilliam77.github.io/RiskAdjustedLinearizations.jl/stable)
[![](https://img.shields.io/badge/docs-latest-3f51b5.svg)](https://chenwilliam77.github.io/RiskAdjustedLinearizations.jl/latest)
[![Master Build Status](https://img.shields.io/travis/chenwilliam77/RiskAdjustedLinearizations.jl?logo=travis)](https://travis-ci.org/chenwilliam77/RiskAdjustedLinearizations.jl)
[![Master Coverage Status](https://coveralls.io/repos/chenwilliam77/RiskAdjustedLinearizations.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/chenwilliam77/RiskAdjustedLinearizations.jl?branch=master)

This package implements [Lopez et al. (2018) "Risk-Adjusted Linearizations of Dynamic Equilibrium Models"](https://ideas.repec.org/p/bfr/banfra/702.html) in Julia. The [original companion code](https://github.com/fvazquezgrande/gen_affine) for the paper implements the method using MATLAB's Symbolic Math Toolbox. RiskAdjustedLinearizations.jl takes advantage of Julia's speed and and flexibility so that the method can be used for solving and estimating large-scale Dynamic Stochatic General Equilibrium (DSGE) models.

## Installation

```julia
pkg> add RiskAdjustedLinearizations
```

The package is compatiable with Julia 1.x.


## Future Development

Please see the [issues](https://github.com/chenwilliam77/RiskAdjustedLinearizations/issues) for additional features planned for implementation.