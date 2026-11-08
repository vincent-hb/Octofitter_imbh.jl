
# PlanetThetaPhasePriorObs — makes the θ parameterisation's phase prior uniform in time

"""
    PlanetThetaPhasePriorObs(epoch_mjd; name)

Jacobian correction that turns a uniform prior on θ (the sky position angle of the
companion at `epoch_mjd`) into a prior that is uniform in time along the orbit, i.e.
uniform in the mean anomaly at that epoch, equivalently uniform in the time of
periastron T₀ (O'Neil et al. 2019, Eq. 30).

When a companion is parameterised by `θ ~ UniformCircular()` and
`tp = θ_at_epoch_to_tperi(θ, epoch_mjd; ...)`, no Jacobian is applied, so the implied
prior density in mean anomaly MA is |dθ/dMA|/2π, with

    dθ/dMA = cos(i) a² √(1−e²) / ρ²

(ρ = projected separation at the epoch; from dθ/dt = h cos i/ρ² and dMA/dt = n,
h = n a² √(1−e²)). This over-weights pericentre passages of eccentric orbits relative
to a time-uniform prior, by up to √(1+e)/(1−e)^{3/2} (≈23 at e = 0.85). The term added
here is

    −log|dθ/dMA| = log ρ² − log|cos i| − 2 log a − ½ log(1−e²),

which integrates to 2π over an orbit, so the marginal priors on a, e and i are
unchanged; only the distribution of phase is corrected. θ remains the sampled variable,
which keeps the sampling geometry of the θ parameterisation.

**`epoch_mjd` must be the same epoch as the one passed to `θ_at_epoch_to_tperi`.**
Use this term only when θ is the sampled phase variable; it is wrong if `tp` (or the
mean anomaly) is sampled directly.

Edge cases: the term is unbounded above as i → 90° when the line of nodes passes
through the companion. The resulting density is integrable (for a generic θ, ρ ∝ |cos i|
near edge-on), but chains should be monitored for pile-up at |cos i| ≪ 1. Degenerate
geometries (ρ = 0, |cos i| < 1e-12, e ≥ 1) return −Inf.

## Example
```julia
epoch = 55927.0
phase = PlanetThetaPhasePriorObs(epoch; name="A_phase")
planet = Planet(
    name = "A",
    basis = Visual{KepOrbit},
    observations = [astrom, pm, phase],
    variables = @variables begin
        ...
        θ ~ UniformCircular()
        tp = θ_at_epoch_to_tperi(θ, \$epoch; a, e, i, ω, Ω, M)
    end
)
```
"""
struct PlanetThetaPhasePriorObs{TTable<:Table} <: AbstractObs
    table::TTable
    priors::Priors
    derived::Derived
    name::String
    function PlanetThetaPhasePriorObs(
            epoch_mjd::Real;
            variables::Tuple{Priors,Derived}=(@variables begin; end),
            name::String
        )
        (priors, derived) = variables
        table = Table(epoch=[Float64(epoch_mjd)])
        return new{typeof(table)}(table, priors, derived, name)
    end
end

export PlanetThetaPhasePriorObs


function likeobj_from_epoch_subset(obs::PlanetThetaPhasePriorObs, obs_inds)
    return PlanetThetaPhasePriorObs(
        obs.table.epoch[obs_inds[1]];
        obs.name,
        variables=(obs.priors, obs.derived,)
    )
end


"""
    _theta_phase_logjac(ρ2, cosi, a, e)

−log|dθ/dMA| = log ρ² − log|cos i| − 2 log a − ½ log(1−e²), with ρ² the squared projected
separation and a the semi-major axis in the same units. Returns −Inf for degenerate
geometries (ρ² ≤ 0, |cos i| < 1e-12, e ≥ 1).
"""
function _theta_phase_logjac(ρ2, cosi, a, e)
    T = promote_type(typeof(ρ2), typeof(cosi), typeof(a), typeof(e))
    abscosi = abs(cosi)
    (ρ2 > 0 && abscosi >= 1e-12 && a > 0 && e < 1) || return T(-Inf)
    return log(ρ2) - log(abscosi) - 2log(a) - log1p(-e^2) / 2
end


function ln_like(obs::PlanetThetaPhasePriorObs, ctx::PlanetObservationContext)
    (; θ_system, orbits, orbit_solutions, i_planet, orbit_solutions_i_epoch_start) = ctx
    T = _system_number_type(θ_system)
    orbit = orbits[i_planet]
    sol = orbit_solutions[i_planet][1 + orbit_solutions_i_epoch_start]
    ρ2 = posx(sol)^2 + posy(sol)^2          # AU²
    a = semimajoraxis(orbit)                # AU
    e = eccentricity(orbit)
    cosi = cos(inclination(orbit))
    return T(_theta_phase_logjac(ρ2, cosi, a, e))
end


# Pure prior term: no data to regenerate.
function generate_from_params(obs::PlanetThetaPhasePriorObs, ctx::PlanetObservationContext; add_noise)
    return obs
end
