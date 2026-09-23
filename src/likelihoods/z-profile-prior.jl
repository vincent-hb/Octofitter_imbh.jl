# PlanetZProfilePriorObs — LOS prior induced by a power-law density profile

"""
    PlanetZProfilePriorObs(epoch_mjd, γ; name)

Prior on the line-of-sight (z) separation of a companion from the central
mass, induced by an isotropic power-law density profile ρ(r) ∝ r^(-γ).

Unlike [`PlanetZPriorObs`](@ref), which applies the same fixed width to every
companion, the width here is set by each companion's own projected separation
R = √(x² + y²) from the central mass, so companions further out in projection
are allowed to sit further out along the line of sight.

## Derivation

For an isotropic population with density ρ(r), the prior on the 3D position is
p(x, y, z) ∝ ρ(√(x² + y² + z²)).  Conditioning on the sky position — so that
this factor constrains only the unobserved coordinate and does not act as a
second prior on the astrometry — gives

```math
p(z \\mid R) = \\frac{\\rho(\\sqrt{R^2 + z^2})}
                     {\\int_{-\\infty}^{\\infty} \\rho(\\sqrt{R^2 + z'^2})\\,dz'}
```

For ρ(r) ∝ r^(-γ) this is a Student-t distribution in z with ν = γ - 1 degrees
of freedom and scale R/√(γ-1):

```math
\\log p(z \\mid R) = \\log C_\\gamma + (\\gamma - 1)\\log R - \\gamma \\log r,
\\qquad
C_\\gamma = \\frac{\\Gamma(\\gamma/2)}{\\sqrt{\\pi}\\,\\Gamma\\!\\left(\\frac{\\gamma-1}{2}\\right)}
```

where r = √(R² + z²) is the 3D separation.

The `(γ-1) log R` normalisation term is **not** optional: R depends on the
fitted orbit (and hence on the fitted position of the central mass), so
omitting it biases the inferred central-mass position toward whichever
configuration maximises the unnormalised profile.

Normalisability requires `γ > 1`.  Note that the tails are heavy — the
variance is finite only for γ > 3 — so for shallow slopes this places only a
weak constraint on z.

## Arguments
- `epoch_mjd`: epoch at which the separation is evaluated.
- `γ`: power-law slope of the density profile.  The Bahcall-Wolf cusp around a
  massive object is γ = 7/4; steeper values are appropriate if the sample
  selection favours small separations.

## Example
```julia
z_prior = PlanetZProfilePriorObs(epoch_mjd, 7/4; name="A_zprior")

planet = Planet(
    name = "A",
    basis = Visual{KepOrbit},
    observations = [astrom, pm, acc, z_prior],
    variables = @variables begin ... end
)
```

See also [`PlanetZPriorObs`](@ref) for the fixed-width Gaussian alternative.
"""
struct PlanetZProfilePriorObs{TTable<:Table} <: AbstractObs
    table::TTable
    γ::Float64
    lognorm::Float64
    priors::Priors
    derived::Derived
    name::String
    function PlanetZProfilePriorObs(
            epoch_mjd::Real,
            γ::Real;
            variables::Tuple{Priors,Derived}=(@variables begin; end),
            name::String
        )
        (priors, derived) = variables
        γ > 1 || throw(ArgumentError(
            "PlanetZProfilePriorObs requires γ > 1 for a normalisable LOS " *
            "prior; got γ = $γ.  For γ ≤ 1 the conditional p(z|R) is improper."))
        γf = Float64(γ)
        # log C_γ = log Γ(γ/2) - log √π - log Γ((γ-1)/2), obtained from the
        # Student-t density at zero to avoid depending on SpecialFunctions:
        # with ν = γ-1, f_t(0) = Γ((ν+1)/2) / (√(νπ) Γ(ν/2)), so C_γ = f_t(0)·√ν.
        # γ is fixed, so this is a Float64 and never enters the AD path.
        lognorm = logpdf(TDist(γf - 1), 0.0) + log(γf - 1) / 2
        table = Table(epoch=[Float64(epoch_mjd)])
        return new{typeof(table)}(table, γf, lognorm, priors, derived, name)
    end
end

export PlanetZProfilePriorObs


function likeobj_from_epoch_subset(obs::PlanetZProfilePriorObs, obs_inds)
    return PlanetZProfilePriorObs(
        obs.table.epoch[obs_inds[1]],
        obs.γ;
        obs.name,
        variables=(obs.priors, obs.derived,)
    )
end


# PlanetZProfilePriorObs likelihood function
function ln_like(zp::PlanetZProfilePriorObs, ctx::PlanetObservationContext)
    (; θ_system, orbit_solutions, i_planet, orbit_solutions_i_epoch_start) = ctx
    T = Octofitter._system_number_type(θ_system)
    sol = orbit_solutions[i_planet][1 + orbit_solutions_i_epoch_start]

    x = posx(sol)
    y = posy(sol)
    z = posz(sol)

    R2 = x^2 + y^2
    r2 = R2 + z^2

    # Degenerate geometry (zero separation) carries no prior mass.
    (R2 > 0 && r2 > 0) || return T(-Inf)

    # log C + (γ-1) log R - γ log r, written with log(R²)/2 and log(r²)/2 to
    # avoid a pair of square roots.
    return T(zp.lognorm) + (zp.γ - 1) / 2 * log(R2) - zp.γ / 2 * log(r2)
end


# Simulation (returns the model z-value and the projected separation that sets
# the width of the prior at that draw)
function simulate(zp::PlanetZProfilePriorObs, θ_system, θ_planet, θ_obs, orbits, orbit_solutions, i_planet, orbit_solutions_i_epoch_start)
    sol = orbit_solutions[i_planet][1 + orbit_solutions_i_epoch_start]
    return (z_model = posz(sol),
            R_model = sqrt(posx(sol)^2 + posy(sol)^2),
            epoch = zp.table.epoch[1])
end


# Generate from params — this is a pure prior, no data to regenerate
function generate_from_params(zp::PlanetZProfilePriorObs, ctx::PlanetObservationContext; add_noise)
    return zp
end
