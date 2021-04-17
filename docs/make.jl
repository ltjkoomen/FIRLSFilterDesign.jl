using Documenter
using FIRLSFilterDesign

makedocs(
    clean = false,
    doctest = false,
    sitename = "FIRLSFilterDesign.jl documentation",
    expandfirst = [],
    modules = [FIRLSFilterDesign],
    pages = [
        "Home" => "index.md",
        "Manual" => "manual.md"
    ],
    )