
# PlanetZPriorObs — Prior on line-of-sight (z) separation from the central mass

"""
    PlanetZPriorObs(epoch_mjd, dist; name)

Prior on the line-of-sight (z) separation of a companion from the central
mass, evaluated at a single epoch.  `dist` is any univariate `Distribution`
(e.g. `Normal(0, σ_z_AU)`).  The model prediction is `posz(sol)` in AU.

This is useful for constraining orbits where the LOS distance is otherwise
unconstrained (e.g. stars in a cluster whose membership restricts how far
they can be from the cluster centre along the line of sight).

## Example
```julia
# σ_z = core radius of cluster in AU
z_prior = PlanetZPriorObs(epoch_mjd, Normal(0.0, 845_000.0); name="A_zprior")

planet = Planet(
    name = "A",
    basis = Visual{KepOrbit},
    observations = [astrom, pm, acc, z_prior],
    variables = @variables begin ... end
)
```
"""
struct PlanetZPriorObs{TTable<:Table, TDist<:Distribution} <: AbstractObs
    table::TTable
    prior_dist::TDist
    priors::Priors
    derived::Derived
    name::String
    function PlanetZPriorObs(
            epoch_mjd::Real,
            dist::Distribution;
            variables::Tuple{Priors,Derived}=(@variables begin; end),
            name::String
        )
        (priors, derived) = variables
        table = Table(epoch=[Float64(epoch_mjd)])
        return new{typeof(table), typeof(dist)}(table, dist, priors, derived, name)
    end
end

export PlanetZPriorObs


function likeobj_from_epoch_subset(obs::PlanetZPriorObs, obs_inds)
    return PlanetZPriorObs(
        obs.table.epoch[obs_inds[1]],
        obs.prior_dist;
        obs.name,
        variables=(obs.priors, obs.derived,)
    )
end


# PlanetZPriorObs likelihood function
function ln_like(zp::PlanetZPriorObs, ctx::PlanetObservationContext)
    (; θ_system, orbit_solutions, i_planet, orbit_solutions_i_epoch_start) = ctx
    T = Octofitter._system_number_type(θ_system)
    sol = orbit_solutions[i_planet][1 + orbit_solutions_i_epoch_start]
    z_au = posz(sol)
    return logpdf(zp.prior_dist, z_au)
end


# Simulation (returns the model z-value)
function simulate(zp::PlanetZPriorObs, θ_system, θ_planet, θ_obs, orbits, orbit_solutions, i_planet, orbit_solutions_i_epoch_start)
    sol = orbit_solutions[i_planet][1 + orbit_solutions_i_epoch_start]
    return (z_model = posz(sol), epoch = zp.table.epoch[1])
end


# Generate from params — this is a pure prior, no data to regenerate
function generate_from_params(zp::PlanetZPriorObs, ctx::PlanetObservationContext; add_noise)
    return zp
end
