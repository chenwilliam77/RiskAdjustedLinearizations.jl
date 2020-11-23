# RiskAdjustedLinearizations.jl

[![GitHub release](https://img.shields.io/github/release/chenwilliam77/RiskAdjustedLinearizations.jl.svg)](https://github.com/chenwilliam77/RiskAdjustedLinearizations.jl/releases/latest)
[![](https://img.shields.io/badge/docs-stable-3f51b5.svg)](https://chenwilliam77.github.io/RiskAdjustedLinearizations.jl/stable)
[![](https://img.shields.io/badge/docs-dev-3f51b5.svg)](https://chenwilliam77.github.io/RiskAdjustedLinearizations.jl/dev)
[![Master Build Status](https://travis-ci.com/chenwilliam77/RiskAdjustedLinearizations.jl.svg?branch=master)](https://travis-ci.com/github/chenwilliam77/RiskAdjustedLinearizations.jl)
[![Coverage Status](https://coveralls.io/repos/github/chenwilliam77/RiskAdjustedLinearizations.jl/badge.svg?branch=master)](https://coveralls.io/github/chenwilliam77/RiskAdjustedLinearizations.jl?branch=master)

This package implements [Lopez et al. (2018) "Risk-Adjusted Linearizations of Dynamic Equilibrium Models"](https://ideas.repec.org/p/bfr/banfra/702.html) in Julia. The [original companion code](https://github.com/fvazquezgrande/gen_affine) for the paper implements the method using MATLAB's Symbolic Math Toolbox. RiskAdjustedLinearizations.jl takes advantage of Julia's speed and flexibility so that the method can be used for solving and estimating large-scale Dynamic Stochastic General Equilibrium (DSGE) models.

Timing tests indicate that this package's speed is significantly faster than the original MATLAB code.
As examples, run the [wac_disaster.jl](https://github.com/chenwilliam77/RiskAdjustedLinearizations.jl/tree/master/examples/matlab_timing_test/wac_disaster.jl) or [rbc_cc.jl](https://github.com/chenwilliam77/RiskAdjustedLinearizations.jl/tree/master/examples/matlab_timing_test/wac_disaster.jl) scripts, which assess how long it takes to calculate a risk-adjusted linearization using the two numerical algorithms
implemented by this package and by the original authors.
The relaxation algorithm in Julia is around 50x-100x faster while the homotopy algorithm in Julia is 3x-4x times faster.

## Installation

```julia
pkg> add RiskAdjustedLinearizations
```

The package is compatiable with Julia `1.x` and is tested in Linux and Windows. The package should also be compatible with MacOS.


## Future Development

Please see the [issues](https://github.com/chenwilliam77/RiskAdjustedLinearizations/issues) for additional features planned for implementation.
