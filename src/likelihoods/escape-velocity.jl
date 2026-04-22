
# PlanetEscVelObs — Häberle-style piecewise escape velocity constraint

"""
    PlanetEscVelObs(epoch_mjd, v2D_kms, σ_v_kms, v_esc_cluster_kms; name)

Piecewise escape velocity constraint following Häberle et al. (2024).

Penalises orbital configurations where the modelled total escape velocity
falls below the observed 2D projected speed of the star.  Bound configurations
(v₂D ≤ v_esc,total) incur no penalty; unbound configurations receive a
Gaussian log-penalty in the velocity excess.

The log-likelihood is:

    ln ℒ = 0                              if  v₂D ≤ v_esc,total
    ln ℒ = -½ ((v₂D - v_esc,total)/σ_v)² if  v₂D > v_esc,total

where

    v_esc,total = √(v_esc,IMBH² + v_esc,cluster²)
    v_esc,IMBH  = √(2GM/r₃D)    [converted to km/s via 1 AU/yr = 4.74047 km/s]
    r₃D         = √(Δα*²/plx² + Δδ²/plx² + z²)   [AU]

and `M` is the IMBH mass from `θ_system.M` [M☉], `plx` is from `θ_system.plx` [mas].

# Arguments
- `epoch_mjd`: Observation epoch [MJD]
- `v2D_kms`:  Observed 2D projected speed [km/s]
- `σ_v_kms`:  Uncertainty on `v2D_kms` [km/s]
- `v_esc_cluster_kms`: Cluster escape velocity at the centre [km/s] (e.g. 62.0 for ω Cen)

# Example
```julia
ev = PlanetEscVelObs(epoch_mjd, star.v2D, star.v2D_err, 62.0; name="A_escvel")

planet = Planet(
    name = "A",
    basis = Visual{KepOrbit},
    observations = [astrom, pm, ev],
    variables = @variables begin ... end
)
```
"""
struct PlanetEscVelObs{TTable<:Table} <: AbstractObs
    table::TTable                # columns: :epoch, :v2D, :σ_v  [km/s]
    v_esc_cluster_kms::Float64   # cluster escape velocity at centre [km/s]
    priors::Priors
    derived::Derived
    name::String
    function PlanetEscVelObs(
            epoch_mjd::Real,
            v2D_kms::Real,
            σ_v_kms::Real,
            v_esc_cluster_kms::Real;
            variables::Tuple{Priors,Derived}=(@variables begin; end),
            name::String
        )
        (priors, derived) = variables
        table = Table(epoch=[Float64(epoch_mjd)], v2D=[Float64(v2D_kms)], σ_v=[Float64(σ_v_kms)])
        return new{typeof(table)}(table, Float64(v_esc_cluster_kms), priors, derived, name)
    end
end

export PlanetEscVelObs


function likeobj_from_epoch_subset(obs::PlanetEscVelObs, obs_inds)
    return PlanetEscVelObs(
        obs.table.epoch[obs_inds[1]],
        obs.table.v2D[obs_inds[1]],
        obs.table.σ_v[obs_inds[1]],
        obs.v_esc_cluster_kms;
        obs.name,
        variables=(obs.priors, obs.derived,)
    )
end


# PlanetEscVelObs likelihood function
function ln_like(obs::PlanetEscVelObs, ctx::PlanetObservationContext)
    (; θ_system, orbit_solutions, i_planet, orbit_solutions_i_epoch_start) = ctx
    T = Octofitter._system_number_type(θ_system)

    sol = orbit_solutions[i_planet][1 + orbit_solutions_i_epoch_start]

    # 3D separation in AU.
    # raoff/decoff are in mas; plx is in mas → raoff/plx is dimensionless (AU/AU = AU).
    plx    = θ_system.plx
    ra_au  = raoff(sol)  / plx
    dec_au = decoff(sol) / plx
    z_au   = posz(sol)
    r_au   = sqrt(ra_au^2 + dec_au^2 + z_au^2)

    # Escape velocity from IMBH alone [km/s].
    # Gaussian gravitational constant: GM = 4π² M  [AU³ yr⁻²] for M in solar masses.
    # v_esc = √(2GM/r) [AU/yr];  1 AU/yr = 4.74047 km/s.
    M = θ_system.M
    v_esc_imbh_kms = oftype(T, 4.74047) * sqrt(oftype(T, 8) * oftype(T, π)^2 * M / r_au)

    # Total escape velocity (IMBH + cluster contributions added in quadrature)
    v_esc_tot = sqrt(v_esc_imbh_kms^2 + oftype(T, obs.v_esc_cluster_kms)^2)

    v2D = oftype(T, obs.table.v2D[1])
    σ_v = oftype(T, obs.table.σ_v[1])

    # Piecewise: no penalty when bound, Gaussian penalty when v₂D exceeds v_esc
    excess = v2D - v_esc_tot
    if excess <= zero(T)
        return zero(T)
    else
        return oftype(T, -0.5) * (excess / σ_v)^2
    end
end


# Simulation: return the model escape velocity and 3D separation
function simulate(obs::PlanetEscVelObs, θ_system, θ_planet, θ_obs, orbits, orbit_solutions, i_planet, orbit_solutions_i_epoch_start)
    sol    = orbit_solutions[i_planet][1 + orbit_solutions_i_epoch_start]
    plx    = θ_system.plx
    ra_au  = raoff(sol)  / plx
    dec_au = decoff(sol) / plx
    z_au   = posz(sol)
    r_au   = sqrt(ra_au^2 + dec_au^2 + z_au^2)
    M      = θ_system.M
    v_esc_imbh_kms = 4.74047 * sqrt(8 * π^2 * M / r_au)
    v_esc_tot      = sqrt(v_esc_imbh_kms^2 + obs.v_esc_cluster_kms^2)
    return (v_esc_model = v_esc_tot, r3D_au = r_au, epoch = obs.table.epoch[1])
end


# Generate from params — observed speed is data, not generated
function generate_from_params(obs::PlanetEscVelObs, ctx::PlanetObservationContext; add_noise)
    return obs
end
