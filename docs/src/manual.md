# Manual

## Installation
To install FIRLS.jl, open up julia and do:
```julia
pkg> update
pkg> add FIRLS
```

## Usage

### Unweighted least-squares FIR design
Let's say we want to design a bandpass filter with a passband from *1 Hz* to *2 Hz*, with a sampling frequency of *6 Hz*, a filter order of *10*, and symmetric filter coefficients. 

We can do this as follows:
```@repl
using FIRLS;
fs = 6;
filter_order = 10;
antisymmetric = false;
```

Set the frequency-band matrix:
```@repl
freq_bands = [0 1; 1 2; 2 3];
```

!!! note
    Frequency bands must not overlap and must completely cover the range ``[0, f_s/2]``.


Define the amplitude response:
```@repl
D = [0. 0.; 1. 1.; 0. 0.];
```

Now we can design the filter:
```@setup A
using FIRLS;
fs = 6;
filter_order = 10;
antisymmetric = false;
freq_bands = [0 1; 1 2; 2 3];
D = [0. 0.; 1. 1.; 0. 0.];
```
```@example A
h = firls_design(filter_order, freq_bands, D, antisymmetric; fs = fs)
```

### Weighted least-squares FIR design
Now let's say we want to design the same filter, but this time we want to place more weight on errors in the passband. We can do this by defining the following weighting coefficient matrix:
```@repl
W = [1 2; 2 2; 2 1];
```

Now we can design the filter with the weight function:
```@setup A
using FIRLS;
fs = 6;
filter_order = 10;
antisymmetric = false;
bands_D = [0 1; 1 2; 2 3];
D = [0. 0.; 1. 1.; 0. 0.];
W = [1 2; 2 2; 2 1];
```
```@example A
h = firls_design(filter_order, bands_D, D, W, antisymmetric; fs = fs)
```

### Different input forms
You can use the `firls_design` function with several different input shapes for the frequency, amplitude response, and weighting function values. They are listed in the table below:

 **Frequency bands**    | **Amplitude response**    | **Weighting function**    | **Comments**                                 
------------------------| --------------------------|---------------------------|-------------------------------------------
 `Matrix`               | `Union{Vector,Matrix}`    | `Union{Vector,Matrix}`    | Vectors are interpreted as constant values over the frequency band.    
 `Vector`               | `Vector`                  | `Vector`                  | Vectors are interpreted as frequency knotpoints and values at those knotpoints.    
 `Matrix`               | `Union{Vector,Matrix}`    | N/A                       | Vectors are interpreted as constant values over the frequency band.   
 `Vector`               | `Vector`                  | N/A                       | Vectors are interpreted as frequency knotpoints and values at those knotpoints.


## Functions
```@autodocs
Modules = [FIRLS]
```