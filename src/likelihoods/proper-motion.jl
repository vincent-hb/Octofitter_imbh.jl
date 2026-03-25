
# PlanetPMObs — Direct 2D proper motion likelihood
const pm_cols = (:epoch, :pmra, :pmdec, :σ_pmra, :σ_pmdec)

"""
    data = Table(
        (epoch = 55197.0, pmra = 3.563, pmdec = 2.564, σ_pmra = 0.038, σ_pmdec = 0.055, cor=0),
    )
    PlanetPMObs(data)

Represents proper motion observations of a body orbiting a central mass.
`:epoch` is a required column (in MJD), along with `:pmra`, `:pmdec`, `:σ_pmra`, `:σ_pmdec`.
All units are in **milliarcseconds per year** (mas/yr).

An optional `:cor` column specifies the correlation between pmra and pmdec errors.

If the system defines `pmra` and/or `pmdec` parameters, they are added to the
orbital proper motion prediction as a systemic offset (e.g. bulk cluster motion).
"""
struct PlanetPMObs{TTable<:Table,TDistTuple} <: AbstractObs
    table::TTable
    priors::Priors
    derived::Derived
    precomputed_pointwise_distributions::TDistTuple
    name::String
    function PlanetPMObs(
            observations;
            variables::Tuple{Priors,Derived}=(@variables begin;end),
            name
        )
        (priors,derived)=variables
        table = Table(observations)
        if !equal_length_cols(table)
            error("The columns in the input data do not all have the same length")
        end
        if !issubset(pm_cols, Tables.columnnames(table))
            error("Expected columns $pm_cols")
        end

        if any(>=(mjd("2050")),  table.epoch) || any(<=(mjd("1950")),  table.epoch)
            @warn "The data you entered fell outside the range year 1950 to year 2050. The expected input format is MJD (modified julian date). We suggest you double check your input data!"
        end

        ii = sortperm(vec(table.epoch))
        table = table[ii]

        # Precompute 2x2 MvNormal distributions for each epoch
        σ₁ = table.σ_pmra
        σ₂ = table.σ_pmdec

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
const PlanetPMLikelihood = PlanetPMObs

export PlanetPMObs, PlanetPMLikelihood


# In-place simulation logic for PlanetPMObs
function simulate!(pmra_model_buf, pmdec_model_buf, pm_obs::PlanetPMObs, θ_system, θ_planet, θ_obs, orbits, orbit_solutions, i_planet, orbit_solutions_i_epoch_start)
    T = _system_number_type(θ_system)

    for i_epoch in eachindex(pm_obs.table.epoch)
        sol = orbit_solutions[i_planet][i_epoch + orbit_solutions_i_epoch_start]
        pmra_model_buf[i_epoch] = pmra(sol)
        pmdec_model_buf[i_epoch] = pmdec(sol)
    end

    return (pmra_model = pmra_model_buf, pmdec_model = pmdec_model_buf, epochs = pm_obs.table.epoch)
end

# Allocating simulation logic
function simulate(pm_obs::PlanetPMObs, θ_system, θ_planet, θ_obs, orbits, orbit_solutions, i_planet, orbit_solutions_i_epoch_start)
    T = _system_number_type(θ_system)
    L = length(pm_obs.table.epoch)
    pmra_model_buf = Vector{T}(undef, L)
    pmdec_model_buf = Vector{T}(undef, L)
    return simulate!(pmra_model_buf, pmdec_model_buf, pm_obs, θ_system, θ_planet, θ_obs, orbits, orbit_solutions, i_planet, orbit_solutions_i_epoch_start)
end


function likeobj_from_epoch_subset(obs::PlanetPMObs, obs_inds)
    return PlanetPMObs(
        obs.table[obs_inds,:,1];
        obs.name,
        variables=(obs.priors, obs.derived,)
    )
end


# PlanetPMObs likelihood function
function ln_like(pm_obs::PlanetPMObs, ctx::PlanetObservationContext)
    (; θ_system, θ_planet, θ_obs, orbits, orbit_solutions, i_planet, orbit_solutions_i_epoch_start) = ctx
    T = Octofitter._system_number_type(θ_system)

    jitter = hasproperty(θ_obs, :jitter) ? getproperty(θ_obs, :jitter) : zero(T)

    L = length(pm_obs.table.epoch)
    ll = zero(T)

    @no_escape begin
        pmra_model_buf = @alloc(T, L)
        pmdec_model_buf = @alloc(T, L)

        Octofitter.simulate!(pmra_model_buf, pmdec_model_buf, pm_obs, θ_system, θ_planet, θ_obs, orbits, orbit_solutions, i_planet, orbit_solutions_i_epoch_start)

        for i_epoch in eachindex(pm_obs.table.epoch)
            pmra_model = pmra_model_buf[i_epoch]
            pmdec_model = pmdec_model_buf[i_epoch]

            # Add system bulk proper motion if defined
            if hasproperty(θ_system, :pmra)
                pmra_model += θ_system.pmra
            end
            if hasproperty(θ_system, :pmdec)
                pmdec_model += θ_system.pmdec
            end

            resid1 = pm_obs.table.pmra[i_epoch] - pmra_model
            resid2 = pm_obs.table.pmdec[i_epoch] - pmdec_model

            if jitter == 0.
                ll += logpdf(pm_obs.precomputed_pointwise_distributions[i_epoch], @SVector[resid1, resid2])
            else
                σ₁ = hypot(pm_obs.table.σ_pmra[i_epoch], jitter)
                σ₂ = hypot(pm_obs.table.σ_pmdec[i_epoch], jitter)
                if hasproperty(pm_obs.table, :cor)
                    cor = pm_obs.table.cor[i_epoch]
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


# Generate new proper motion observations from model
function generate_from_params(like::PlanetPMObs, ctx::PlanetObservationContext; add_noise)
    (; θ_system, θ_planet, θ_obs, orbits, orbit_solutions, i_planet, orbit_solutions_i_epoch_start) = ctx

    epoch = like.table.epoch

    sim = Octofitter.simulate(like, θ_system, θ_planet, θ_obs, orbits, orbit_solutions, i_planet, orbit_solutions_i_epoch_start)
    pmra_model = collect(sim.pmra_model)
    pmdec_model = collect(sim.pmdec_model)

    # Add system bulk PM if defined
    if hasproperty(θ_system, :pmra)
        pmra_model .+= θ_system.pmra
    end
    if hasproperty(θ_system, :pmdec)
        pmdec_model .+= θ_system.pmdec
    end

    jitter = hasproperty(θ_obs, :jitter) ? getproperty(θ_obs, :jitter) : 0.0

    σ_pmra = like.table.σ_pmra
    σ_pmdec = like.table.σ_pmdec

    pmra_out = pmra_model
    pmdec_out = pmdec_model

    if add_noise
        pmra_out .+= randn.() .* hypot.(σ_pmra, jitter)
        pmdec_out .+= randn.() .* hypot.(σ_pmdec, jitter)
    end

    if hasproperty(like.table, :cor)
        pm_table = Table(; epoch, pmra=pmra_out, pmdec=pmdec_out, σ_pmra, σ_pmdec, like.table.cor)
    else
        pm_table = Table(; epoch, pmra=pmra_out, pmdec=pmdec_out, σ_pmra, σ_pmdec)
    end

    return PlanetPMObs(pm_table; like.name)
end
