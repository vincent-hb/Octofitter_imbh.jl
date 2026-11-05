
# PlanetAccelObs — Direct 2D acceleration likelihood
const accel_cols = (:epoch, :accra, :accdec, :σ_accra, :σ_accdec)

"""
    data = Table(
        (epoch = 55197.0, accra = -0.0069, accdec = 0.0085, σ_accra = 0.0083, σ_accdec = 0.0098, cor=0),
    )
    PlanetAccelObs(data)

Represents angular acceleration observations of a body orbiting a central mass.
`:epoch` is a required column (in MJD), along with `:accra`, `:accdec`, `:σ_accra`, `:σ_accdec`.
All units are in **milliarcseconds per year squared** (mas/yr²).

An optional `:cor` column specifies the correlation between accra and accdec errors.

Three optional **system-level** calibration parameters are read from `θ_system`
when they are declared in the `System` `@variables` block; each defaults to zero
and is shared by every body in the system (the acceleration measurements of all
bodies come from the same astrometric reduction):

- `acc_bias_ra`, `acc_bias_dec` [mas/yr²]: a common zero point of the measured
  accelerations, subtracted from `accra`/`accdec` before the residual is formed
  (`resid = accra_data - acc_bias_ra - accra_model`). With Gaussian priors on
  both, the fit marginalizes over a calibration offset that would otherwise be
  absorbed by the central mass and the line-of-sight distances.
- `acc_jitter` [mas/yr²]: extra scatter added in quadrature to `σ_accra` and
  `σ_accdec` of every body.

An observation-level `jitter` (declared in this observation's own `@variables`
block) is also supported and is combined in quadrature with `acc_jitter`.
Note that the zero point applies to this vector likelihood only;
`PlanetAccelMagObs` carries no direction and ignores `acc_bias_*`.
"""
struct PlanetAccelObs{TTable<:Table,TDistTuple} <: AbstractObs
    table::TTable
    priors::Priors
    derived::Derived
    precomputed_pointwise_distributions::TDistTuple
    name::String
    function PlanetAccelObs(
            observations;
            variables::Tuple{Priors,Derived}=(@variables begin;end),
            name
        )
        (priors,derived)=variables
        table = Table(observations)
        if !equal_length_cols(table)
            error("The columns in the input data do not all have the same length")
        end
        if !issubset(accel_cols, Tables.columnnames(table))
            error("Expected columns $accel_cols")
        end

        if any(>=(mjd("2050")),  table.epoch) || any(<=(mjd("1950")),  table.epoch)
            @warn "The data you entered fell outside the range year 1950 to year 2050. The expected input format is MJD (modified julian date). We suggest you double check your input data!"
        end

        ii = sortperm(vec(table.epoch))
        table = table[ii]

        # Precompute 2x2 MvNormal distributions for each epoch
        σ₁ = table.σ_accra
        σ₂ = table.σ_accdec

        if hasproperty(table, :cor)
            cor = table.cor

            if any(abs.(cor) .> 1 - 1e-5)
                error("Correlation values may not be well-specified: $cor")
            end

            precomputed_pointwise_distributions = broadcast(σ₁, σ₂, cor) do σ₁, σ₂, cor
                Σ = @SArray[
                    σ₁^2        cor*σ₁*σ₂
                    cor*σ₁*σ₂   σ₂^2
                ]
                dist = MvNormal(Σ)
                return dist
            end
        else
            precomputed_pointwise_distributions = broadcast(σ₁, σ₂) do σ₁, σ₂
                Σ = Diagonal(@SArray[σ₁^2, σ₂^2])
                dist = MvNormal(Σ)
                return dist
            end
        end

        precomputed_pointwise_distributions_tuple = (precomputed_pointwise_distributions...,)
        return new{typeof(table),typeof(precomputed_pointwise_distributions_tuple)}(
            table, priors, derived, precomputed_pointwise_distributions_tuple, name
        )
    end
end

# Backwards compatibility alias
const PlanetAccelLikelihood = PlanetAccelObs

export PlanetAccelObs, PlanetAccelLikelihood


# In-place simulation logic for PlanetAccelObs
function simulate!(accra_model_buf, accdec_model_buf, acc_obs::PlanetAccelObs, θ_system, θ_planet, θ_obs, orbits, orbit_solutions, i_planet, orbit_solutions_i_epoch_start)
    T = _system_number_type(θ_system)

    for i_epoch in eachindex(acc_obs.table.epoch)
        sol = orbit_solutions[i_planet][i_epoch + orbit_solutions_i_epoch_start]
        accra_model_buf[i_epoch] = accra(sol)
        accdec_model_buf[i_epoch] = accdec(sol)
    end

    return (accra_model = accra_model_buf, accdec_model = accdec_model_buf, epochs = acc_obs.table.epoch)
end

# Allocating simulation logic
function simulate(acc_obs::PlanetAccelObs, θ_system, θ_planet, θ_obs, orbits, orbit_solutions, i_planet, orbit_solutions_i_epoch_start)
    T = _system_number_type(θ_system)
    L = length(acc_obs.table.epoch)
    accra_model_buf = Vector{T}(undef, L)
    accdec_model_buf = Vector{T}(undef, L)
    return simulate!(accra_model_buf, accdec_model_buf, acc_obs, θ_system, θ_planet, θ_obs, orbits, orbit_solutions, i_planet, orbit_solutions_i_epoch_start)
end


function likeobj_from_epoch_subset(obs::PlanetAccelObs, obs_inds)
    return PlanetAccelObs(
        obs.table[obs_inds,:,1];
        obs.name,
        variables=(obs.priors, obs.derived,)
    )
end


# PlanetAccelObs likelihood function
function ln_like(acc_obs::PlanetAccelObs, ctx::PlanetObservationContext)
    (; θ_system, θ_planet, θ_obs, orbits, orbit_solutions, i_planet, orbit_solutions_i_epoch_start) = ctx
    T = Octofitter._system_number_type(θ_system)

    jitter = hasproperty(θ_obs, :jitter) ? getproperty(θ_obs, :jitter) : zero(T)
    # System-level calibration terms shared by all bodies (see the docstring);
    # zero unless declared, so models without them are unchanged.  Read from
    # θ_system like offsetx/offsety in relative-astrometry.jl.
    acc_bias_ra  = hasproperty(θ_system, :acc_bias_ra)  ? getproperty(θ_system, :acc_bias_ra)  : zero(T)
    acc_bias_dec = hasproperty(θ_system, :acc_bias_dec) ? getproperty(θ_system, :acc_bias_dec) : zero(T)
    acc_jitter   = hasproperty(θ_system, :acc_jitter)   ? getproperty(θ_system, :acc_jitter)   : zero(T)

    L = length(acc_obs.table.epoch)
    ll = zero(T)

    @no_escape begin
        accra_model_buf = @alloc(T, L)
        accdec_model_buf = @alloc(T, L)

        Octofitter.simulate!(accra_model_buf, accdec_model_buf, acc_obs, θ_system, θ_planet, θ_obs, orbits, orbit_solutions, i_planet, orbit_solutions_i_epoch_start)

        for i_epoch in eachindex(acc_obs.table.epoch)
            accra_model = accra_model_buf[i_epoch]
            accdec_model = accdec_model_buf[i_epoch]

            resid1 = acc_obs.table.accra[i_epoch] - acc_bias_ra - accra_model
            resid2 = acc_obs.table.accdec[i_epoch] - acc_bias_dec - accdec_model

            if jitter == 0. && acc_jitter == 0.
                ll += logpdf(acc_obs.precomputed_pointwise_distributions[i_epoch], @SVector[resid1, resid2])
            else
                # Squares rather than hypot: the derivative of hypot(σ, j) is
                # undefined at j = 0, which a jitter prior may touch.
                σ₁ = sqrt(acc_obs.table.σ_accra[i_epoch]^2 + jitter^2 + acc_jitter^2)
                σ₂ = sqrt(acc_obs.table.σ_accdec[i_epoch]^2 + jitter^2 + acc_jitter^2)
                if hasproperty(acc_obs.table, :cor)
                    cor = acc_obs.table.cor[i_epoch]
                    Σ = @SArray[
                        σ₁^2        cor*σ₁*σ₂
                        cor*σ₁*σ₂   σ₂^2
                    ]
                    dist = MvNormal(Σ)
                else
                    Σ = Diagonal(@SArray[σ₁^2, σ₂^2])
                    dist = MvNormal(Σ)
                end
                ll += logpdf(dist, @SVector[resid1, resid2])
            end
        end
    end
    return ll
end


# Generate new acceleration observations from model
function generate_from_params(like::PlanetAccelObs, ctx::PlanetObservationContext; add_noise)
    (; θ_system, θ_planet, θ_obs, orbits, orbit_solutions, i_planet, orbit_solutions_i_epoch_start) = ctx

    epoch = like.table.epoch

    sim = Octofitter.simulate(like, θ_system, θ_planet, θ_obs, orbits, orbit_solutions, i_planet, orbit_solutions_i_epoch_start)
    accra_model = collect(sim.accra_model)
    accdec_model = collect(sim.accdec_model)

    jitter = hasproperty(θ_obs, :jitter) ? getproperty(θ_obs, :jitter) : 0.0
    # System-level calibration terms, as in ln_like: a generated data set
    # carries the same zero point and extra scatter the model was fitted with.
    acc_bias_ra  = hasproperty(θ_system, :acc_bias_ra)  ? θ_system.acc_bias_ra  : 0.0
    acc_bias_dec = hasproperty(θ_system, :acc_bias_dec) ? θ_system.acc_bias_dec : 0.0
    acc_jitter   = hasproperty(θ_system, :acc_jitter)   ? θ_system.acc_jitter   : 0.0

    σ_accra = like.table.σ_accra
    σ_accdec = like.table.σ_accdec

    accra_out = accra_model .+ acc_bias_ra
    accdec_out = accdec_model .+ acc_bias_dec

    if add_noise
        accra_out .+= randn.() .* sqrt.(σ_accra.^2 .+ jitter^2 .+ acc_jitter^2)
        accdec_out .+= randn.() .* sqrt.(σ_accdec.^2 .+ jitter^2 .+ acc_jitter^2)
    end

    if hasproperty(like.table, :cor)
        accel_table = Table(; epoch, accra=accra_out, accdec=accdec_out, σ_accra, σ_accdec, like.table.cor)
    else
        accel_table = Table(; epoch, accra=accra_out, accdec=accdec_out, σ_accra, σ_accdec)
    end

    return PlanetAccelObs(accel_table; like.name)
end
