# FIRLS.jl - Least-squares FIR filter design
FIRLS.jl is a julia package to perform linear-phase FIR filter design using weighted least-squares. It supports the following:
- Piecewise linear amplitude response functions
- Piecewise linear weighting functions
- Type I, II, III, IV FIR filters
- Custom linear solver function

## Installation
To install FIRLS.jl, open up a Julia REPL and do:
```@repl
using Pkg
Pkg.update()
Pkg.add("FIRLS")
```