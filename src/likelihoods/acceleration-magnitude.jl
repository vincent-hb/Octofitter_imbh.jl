
# PlanetAccelMagObs — Sky-plane acceleration magnitude likelihood
const accel_mag_cols = (:epoch, :accmag, :σ_accmag)

"""
    data = Table(
        epoch    = [55197.0],
        accmag   = [0.011],
        σ_accmag = [0.013],
    )
    PlanetAccelMagObs(data; name="star_accmag")

Represents a constraint on the sky-plane acceleration **magnitude**
|a_sky| = √(aα*² + aδ²) of a body orbiting a central mass, without any
directional information.

`:epoch` is required (MJD), along with `:accmag` and `:σ_accmag` (both in mas/yr²).

Use this in place of `PlanetAccelObs` when the direction of the measured
acceleration should not constrain the IMBH position — only the separation
matters.  The model prediction is `|a_model| = √(accra(sol)² + accdec(sol)²)`.

The likelihood is a scalar Normal evaluated at `|a_obs| − |a_model|`, which
is the Gaussian approximation to a Rice distribution; it is accurate when
SNR ≡ |a_obs| / σ_accmag ≳ 2.  The uncertainty `σ_accmag` should be
propagated from the vector components via
    σ_|a| = √( (aα* σ_aα*)² + (aδ σ_aδ)² ) / |a|.
`octo_utils.build_star_observations` does this automatically when
`acceleration_type = "magnitude"`.
"""
struct PlanetAccelMagObs{TTable<:Table,TDistTuple} <: AbstractObs
    table::TTable
    priors::Priors
    derived::Derived
    precomputed_pointwise_distributions::TDistTuple
    name::String
    function PlanetAccelMagObs(
            observations;
            variables::Tuple{Priors,Derived}=(@variables begin;end),
            name
        )
        (priors, derived) = variables
        table = Table(observations)
        if !equal_length_cols(table)
            error("The columns in the input data do not all have the same length")
        end
        if !issubset(accel_mag_cols, Tables.columnnames(table))
            error("Expected columns $accel_mag_cols")
        end

        if any(>=(mjd("2050")), table.epoch) || any(<=(mjd("1950")), table.epoch)
            @warn "The data you entered fell outside the range year 1950 to year 2050. " *
                  "The expected input format is MJD (modified julian date). " *
                  "We suggest you double check your input data!"
        end

        ii = sortperm(vec(table.epoch))
        table = table[ii]

        # Precompute scalar Normal(0, σ_accmag) per epoch (zero-centred; residual is passed in)
        precomputed_pointwise_distributions = Tuple(
            Normal(zero(Float64), Float64(σ)) for σ in table.σ_accmag
        )

        return new{typeof(table), typeof(precomputed_pointwise_distributions)}(
            table, priors, derived, precomputed_pointwise_distributions, name
        )
    end
end

export PlanetAccelMagObs


# In-place simulation: fill buffer with |a_sky| = √(accra² + accdec²) at each epoch
function simulate!(accmag_model_buf, acc_mag_obs::PlanetAccelMagObs,
                   θ_system, θ_planet, θ_obs, orbits, orbit_solutions,
                   i_planet, orbit_solutions_i_epoch_start)
    for i_epoch in eachindex(acc_mag_obs.table.epoch)
        sol = orbit_solutions[i_planet][i_epoch + orbit_solutions_i_epoch_start]
        accmag_model_buf[i_epoch] = sqrt(accra(sol)^2 + accdec(sol)^2)
    end
    return (accmag_model = accmag_model_buf, epochs = acc_mag_obs.table.epoch)
end

# Allocating simulation
function simulate(acc_mag_obs::PlanetAccelMagObs,
                  θ_system, θ_planet, θ_obs, orbits, orbit_solutions,
                  i_planet, orbit_solutions_i_epoch_start)
    T = _system_number_type(θ_system)
    L = length(acc_mag_obs.table.epoch)
    accmag_model_buf = Vector{T}(undef, L)
    return simulate!(accmag_model_buf, acc_mag_obs, θ_system, θ_planet, θ_obs,
                     orbits, orbit_solutions, i_planet, orbit_solutions_i_epoch_start)
end


function likeobj_from_epoch_subset(obs::PlanetAccelMagObs, obs_inds)
    return PlanetAccelMagObs(
        obs.table[obs_inds, :, 1];
        obs.name,
        variables = (obs.priors, obs.derived)
    )
end


# Log-likelihood
function ln_like(acc_mag_obs::PlanetAccelMagObs, ctx::PlanetObservationContext)
    (; θ_system, θ_planet, θ_obs, orbits, orbit_solutions,
       i_planet, orbit_solutions_i_epoch_start) = ctx
    T = Octofitter._system_number_type(θ_system)

    jitter = hasproperty(θ_obs, :jitter) ? getproperty(θ_obs, :jitter) : zero(T)

    L  = length(acc_mag_obs.table.epoch)
    ll = zero(T)

    @no_escape begin
        accmag_model_buf = @alloc(T, L)

        simulate!(accmag_model_buf, acc_mag_obs, θ_system, θ_planet, θ_obs,
                  orbits, orbit_solutions, i_planet, orbit_solutions_i_epoch_start)

        for i_epoch in eachindex(acc_mag_obs.table.epoch)
            accmag_model = accmag_model_buf[i_epoch]
            resid = acc_mag_obs.table.accmag[i_epoch] - accmag_model

            if jitter == 0.
                ll += logpdf(acc_mag_obs.precomputed_pointwise_distributions[i_epoch], resid)
            else
                σ = hypot(acc_mag_obs.table.σ_accmag[i_epoch], jitter)
                ll += logpdf(Normal(zero(T), σ), resid)
            end
        end
    end
    return ll
end


# Posterior predictive sample
function generate_from_params(like::PlanetAccelMagObs, ctx::PlanetObservationContext; add_noise)
    (; θ_system, θ_planet, θ_obs, orbits, orbit_solutions,
       i_planet, orbit_solutions_i_epoch_start) = ctx

    epoch    = like.table.epoch
    σ_accmag = like.table.σ_accmag

    sim = simulate(like, θ_system, θ_planet, θ_obs, orbits, orbit_solutions,
                   i_planet, orbit_solutions_i_epoch_start)
    accmag_out = collect(sim.accmag_model)

    jitter = hasproperty(θ_obs, :jitter) ? getproperty(θ_obs, :jitter) : 0.0

    if add_noise
        accmag_out .+= randn.() .* hypot.(σ_accmag, jitter)
    end

    return PlanetAccelMagObs(Table(; epoch, accmag = accmag_out, σ_accmag); like.name)
end
