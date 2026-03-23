# Ocen_IMBH Project

## Project Purpose

This project fits orbits of stars around an intermediate-mass black hole (IMBH) candidate in the globular cluster ω Centauri (ω Cen). We use a customized development fork of [Octofitter](https://github.com/sefffal/Octofitter.jl) (`Octofitter_imbh.jl`) to perform Bayesian inference on stellar kinematics.

**Key development goals in `Octofitter_imbh.jl`:**
- Enable the RA and DEC position of the central mass to be free parameters in the model
- Implement a likelihood for fitting directly on 2D proper motion velocities and accelerations (RA and DEC components separately, not just the scalar anomaly magnitude)

**Analysis and compute (`Ocen_IMBH_analysis/`):**
- Scripts to launch Slurm jobs for orbit fitting on Digital Alliance of Canada (DAC) HPC clusters
- Scripts for post-fit analysis, chain diagnostics, and figure production

---

## Codebase Summary

### `Octofitter_imbh.jl/` — Development fork of Octofitter (v8.1.2, Julia)

A Bayesian orbital fitting framework. Users define a `System` (with `Planet` objects and `AbstractObs` likelihood observations), compile a `LogDensityModel`, then sample via HMC/NUTS using AdvancedHMC.jl. Gradients are computed automatically via ForwardDiff.jl.

**Core source (`src/`):**
- `Octofitter.jl` — module entry point and re-exports
- `variables.jl` — `System`, `Planet`, `@variables` macro, priors, derived parameters
- `logdensitymodel.jl` — `LogDensityModel` struct (implements `LogDensityProblems` interface)
- `sampling.jl` — HMC/NUTS sampling, initialization, chain management
- `initialization.jl` — heuristic parameter initialization
- `parameterizations.jl` — parameter transformations (e.g., `UniformCircular`)
- `distributions.jl` — custom distributions (`Sine`, `UniformImproper`, `KDEDist`)
- `analysis.jl` — stubs that dispatch to Makie extension functions
- `likelihoods/` — one file per data type (`relative-astrometry.jl`, `hgca.jl`, `gaia-dr4.jl`, `hipparcos.jl`, `photometry.jl`, etc.)

**Extension packages (optional, separate `Project.toml`):**
- `OctofitterRadialVelocity/` — absolute and relative RV, Celerite GP kernel
- `OctofitterInterferometry/` — GRAVITY interferometric data
- `OctofitterImages/` — direct imaging contrast maps
- `OctofitterTransits/` — transit photometry

**Makie extension (`ext/OctofitterMakieExt/`):**
- One file per plot type (`astromplot.jl`, `hgcaplot.jl`, `pmaplot.jl`, `rvtimeplot.jl`, `gaiastarplot.jl`, `dotplot.jl`, `octoplot.jl`, etc.)
- Utility functions in `util.jl` (`_date_ticks`, `concat_with_nan`, etc.)

### `Ocen_IMBH_analysis/` — Analysis scripts

Currently minimal. Will contain:
- Slurm job submission scripts for DAC clusters
- Julia/Python scripts for chain diagnostics and posterior analysis
- Figure production scripts

---

## Octofitter Coding Conventions

### Naming
- Functions: `snake_case` (e.g., `octoplot`, `construct_elements`)
- Mutating functions: `!` suffix (e.g., `octoplot!`, `astromplot!`)
- Private/internal: `_` prefix (e.g., `_date_ticks`, `_system_number_type`)
- Types/structs: `PascalCase` (e.g., `LogDensityModel`, `AbstractObs`, `SystemObservationContext`)
- Fields and local variables: `snake_case`
- Mathematical variables: Greek letters and Unicode encouraged (e.g., `θ`, `ℓ`, `∇ℓπ`, `α`, `ν`)

### Types and Structs
- Use abstract types for extensibility (e.g., `AbstractObs`, `AbstractOrbit`)
- Parametric structs with `<:` constraints for type stability
- Use `NamedTuple` for parameter bundles (e.g., `θ_system`, `θ_planet`, `θ_obs`)
- Prefer immutable structs for data; mutable only for stateful objects (e.g., `LogDensityModel`)

### Model Definition Patterns
- Use the `@variables` DSL macro with `begin...end` blocks for defining priors and derived quantities
- Observations/likelihoods are subtypes of `AbstractObs`; new likelihood types go in `src/likelihoods/`
- The `LogDensityModel(system)` call compiles the model (generates log-likelihood and gradient code)

### Automatic Differentiation
- All likelihood and model code must be AD-compatible (ForwardDiff by default)
- Avoid non-differentiable branches in likelihood evaluation paths
- Type stability is critical: use `let` blocks in closures to capture variables, avoid untyped globals
- Do not use `Float64` literals where the element type should propagate from inputs

### Docstrings
- Triple-quoted `"""..."""` above function/type definitions
- First line: concise one-sentence summary
- Follow with parameter descriptions and usage examples
- Use LaTeX/Unicode for mathematical notation

---

## Plot Formatting Conventions (Makie / CairoMakie)

### Figure Sizes and Saving
- Main figures: `size=(700, 600)`
- Compact panels: `size=(500, 300)` to `(500, 400)`
- Always save with `px_per_unit=3` for publication quality
- Accept a `figure=(;)` keyword argument to allow caller overrides

### Colors
- Per-object colormap: `Makie.cgrad([Makie.wong_colors()[i], "#FAFAFA"])`
- For multiple objects: cycle through `Makie.wong_colors()` by index
- Specific-use colormaps: `:plasma` for RV time, `:turbo` for eccentricity scatter, `:Egypt`/`:Lakota` for categorical instrument/epoch separation

### Axes
- No grid lines: `xgridvisible=false, ygridvisible=false`
- Astrometry plots: `autolimitaspect=1` (square), `xreversed=true` (RA increases right-to-left)
- Zero-crossing reference: `vlines!(ax, 0, color=:grey, linestyle=:dash)`

### Axis Labels
- Rich text for sub/superscripts: `Makie.rich("μ", Makie.subscript("α*"), " [mas/yr]")`
- Mathematical notation: `Makie.latexstring(...)` where needed
- Unicode deltas in coordinate labels: Δα*, Δδ

### Markers and Lines
- Data points: `markersize=8`
- Posterior sample scatter: `markersize=2`, adaptive alpha `alpha=min.(1, 100/length(ii))`
- Central mass / star marker: `marker='★', markersize=20, color=:white, strokecolor=:black, strokewidth=1.5`
- Dense scatter: `rasterize=4` to reduce output file size

### Function Pattern
Every plot type exposes two methods:
```julia
# Standalone: creates Figure, saves to file, returns Figure
function Octofitter.myplot(model, results, fname="$(model.system.name)-myplot.png"; figure=(;), kwargs...)
    fig = Figure(; size=(...), figure...)
    Octofitter.myplot!(fig.layout, model, results; kwargs...)
    Makie.save(fname, fig, px_per_unit=3)
    return fig
end

# In-place: draws into an existing GridLayout
function Octofitter.myplot!(layout, model, results; kwargs...)
    ax = Axis(layout[1,1]; ...)
    # ...
end
```
