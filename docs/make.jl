using Documenter
using FIRLS

makedocs(
    clean = false,
    doctest = false,
    sitename = "FIRLS.jl documentation",
    expandfirst = [],
    modules = [FIRLS],
    pages = [
        "Home" => "index.md",
        "Manual" => "manual.md"
    ],
    )