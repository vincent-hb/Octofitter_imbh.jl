"""
    MixtureGaussianReference(means, stds; weights=nothing, first_round=1)

Fixed mixture of diagonal Gaussians in the model's unconstrained (linked) parameter
space, used as the variational reference of stabilized parallel tempering
(`octofit_pigeons(model; variational=MixtureGaussianReference(...))`).

Use it when the posterior has well-separated modes. The default Pigeons variational
reference (`GaussianReference`) is one diagonal Gaussian refitted every round to the
target chain's samples. Once the target chain holds one mode, that Gaussian fits
that mode, and the variational leg keeps feeding restarts into it. A fixed reference
with one component per mode feeds every mode. The component locations come from
earlier runs (`mixture_reference_from_chains`).

The reference only affects efficiency. Parallel tempering is exact for any reference,
so the target's mode weights are set by the swap acceptances, not by `weights`; equal
weights are the default. The components need to cover the modes, not match them.

Density, with ``K`` components, weights ``w_k``, means ``μ_k`` and standard deviations
``σ_k`` (vectors of length ``D``):
```math
q(x) = \\sum_k w_k \\prod_i \\mathcal{N}(x_i; μ_{k,i}, σ_{k,i}^2)
```

Arguments:
- `means`, `stds`: `D × K` matrices, or vectors of `K` length-`D` vectors, in linked
  space (`model.link(θ)`), in the flat parameter order of the model.
- `weights`: `K` positive numbers, normalised internally (default: equal).
- `first_round`: Pigeons activates the reference after this round, i.e. it is used
  from round `first_round + 1`. Before that, the variational leg uses the prior.

The Pigeons interface (`activate_variational`, `sample_iid!`, ...) is implemented in
the Pigeons extension. With Pigeons loaded, the object is also a `LogDensityProblems`
log density with an analytic gradient, for gradient-based explorers.
"""
struct MixtureGaussianReference
    "log weights, normalised (logsumexp = 0)"
    logweights::Vector{Float64}
    "D × K component means (linked space)"
    means::Matrix{Float64}
    "D × K component standard deviations (linked space)"
    stds::Matrix{Float64}
    "per-component normalising constants Σᵢ −½ log(2π σᵢ²)"
    lognorms::Vector{Float64}
    "Pigeons activates the reference after this round"
    first_round::Int
    function MixtureGaussianReference(means::AbstractMatrix, stds::AbstractMatrix;
                                      weights=nothing, first_round::Integer=1)
        size(means) == size(stds) || error("means and stds must have the same size (got $(size(means)) and $(size(stds)))")
        D, K = size(means)
        D ≥ 1 && K ≥ 1 || error("need at least one dimension and one component")
        all(isfinite, means) || error("non-finite component mean")
        all(s -> isfinite(s) && s > 0, stds) || error("component standard deviations must be finite and > 0")
        w = isnothing(weights) ? fill(1.0, K) : collect(Float64, weights)
        length(w) == K || error("weights has length $(length(w)), expected $K (one per component)")
        all(x -> isfinite(x) && x > 0, w) || error("weights must be finite and > 0")
        first_round ≥ 1 || error("first_round must be ≥ 1")
        logw = log.(w ./ sum(w))
        lognorms = [-sum(log, view(stds, :, k)) - D * log(2π) / 2 for k in 1:K]
        return new(logw, Matrix{Float64}(means), Matrix{Float64}(stds), lognorms, Int(first_round))
    end
end
MixtureGaussianReference(means::AbstractVector{<:AbstractVector}, stds::AbstractVector{<:AbstractVector}; kwargs...) =
    MixtureGaussianReference(reduce(hcat, means), reduce(hcat, stds); kwargs...)
export MixtureGaussianReference

_mixture_dim(ref::MixtureGaussianReference) = size(ref.means, 1)
_mixture_ncomp(ref::MixtureGaussianReference) = size(ref.means, 2)

# log of each weighted component density at x (length K)
function _mixture_component_logdens(ref::MixtureGaussianReference, x::AbstractVector)
    D, K = size(ref.means)
    length(x) == D || error("state has length $(length(x)), reference has dimension $D")
    out = Vector{Float64}(undef, K)
    @inbounds for k in 1:K
        s = 0.0
        for i in 1:D
            z = (x[i] - ref.means[i, k]) / ref.stds[i, k]
            s += z * z
        end
        out[k] = ref.logweights[k] + ref.lognorms[k] - s / 2
    end
    return out
end

function _logsumexp_vec(v::AbstractVector)
    m = maximum(v)
    isfinite(m) || return m
    return m + log(sum(x -> exp(x - m), v))
end

"""
    (ref::MixtureGaussianReference)(x)

Log density of the mixture reference at the linked parameter vector `x`.
"""
(ref::MixtureGaussianReference)(x::AbstractVector) = _logsumexp_vec(_mixture_component_logdens(ref, x))

"""
    _mixture_logdensity_and_gradient!(grad, ref, x)

Log density and its gradient (written into `grad`): ∇ log q = Σₖ rₖ(x) (μₖ − x)/σₖ²,
with rₖ the component responsibilities.
"""
function _mixture_logdensity_and_gradient!(grad::AbstractVector, ref::MixtureGaussianReference, x::AbstractVector)
    D, K = size(ref.means)
    lc = _mixture_component_logdens(ref, x)
    l = _logsumexp_vec(lc)
    fill!(grad, 0.0)
    @inbounds for k in 1:K
        r = exp(lc[k] - l)
        r == 0 && continue
        for i in 1:D
            grad[i] += r * (ref.means[i, k] - x[i]) / ref.stds[i, k]^2
        end
    end
    return l, grad
end

"""
    _mixture_draw!(rng, x, ref)

Overwrite `x` with an exact draw from the mixture: a component by weight, then each
coordinate from that component's Gaussian.
"""
function _mixture_draw!(rng::AbstractRNG, x::AbstractVector, ref::MixtureGaussianReference)
    D, K = size(ref.means)
    length(x) == D || error("state has length $(length(x)), reference has dimension $D")
    u = rand(rng)
    k = K
    c = 0.0
    for j in 1:K
        c += exp(ref.logweights[j])
        if u < c
            k = j
            break
        end
    end
    @inbounds for i in 1:D
        x[i] = ref.means[i, k] + ref.stds[i, k] * randn(rng)
    end
    return x
end

"""
    _flat_parameter_names(system::System)

Names of the sampled parameters in the order of the flat parameter vector
(`_list_priors` / `make_arr2nt`), as they appear in chains: system variables by name,
planet variables as `<planet>_<variable>` (`UniformCircular` pairs as e.g. `A_θx`,
`A_θy`). Observation-level variables are not supported (error).
"""
function _flat_parameter_names(system::System)
    names = Symbol[]
    for k in keys(system.priors.priors)
        push!(names, Symbol(k))
    end
    for obs in system.observations
        if hasproperty(obs, :priors) && !isempty(obs.priors.priors)
            error("observation-level variables (system observation of type $(nameof(typeof(obs)))) are not supported by mixture_reference_from_chains")
        end
    end
    for planet in system.planets
        for k in keys(planet.priors.priors)
            push!(names, Symbol(planet.name, '_', k))
        end
        for obs in planet.observations
            if hasproperty(obs, :priors) && !isempty(obs.priors.priors)
                error("observation-level variables (planet $(planet.name), observation of type $(nameof(typeof(obs)))) are not supported by mixture_reference_from_chains")
            end
        end
    end
    return names
end

"""
    linked_samples(model, chain; ii=:)

Matrix (`D × N`) of the posterior draws of `chain` mapped to the linked (unconstrained)
parameter space of `model`, columns in the model's flat parameter order. Columns are
looked up by name, so the chain may contain extra parameters (e.g. a chain of a model
with more stars). Draws that fall outside the support of `model`'s priors (non-finite
after linking) are dropped with a warning.
"""
function linked_samples(model::LogDensityModel, chain::MCMCChains.Chains; ii=Colon())
    names = _flat_parameter_names(model.system)
    length(names) == model.D || error("internal: $(length(names)) parameter names for D = $(model.D)")
    chain_names = Set(Symbol.(MCMCChains.names(chain)))
    missing_names = filter(n -> n ∉ chain_names, names)
    isempty(missing_names) || error("chain is missing parameters of this model: $(join(missing_names, ", "))")
    cols = [vec(Array(chain[n]))[ii] for n in names]
    N = length(first(cols))
    X = Matrix{Float64}(undef, model.D, N)
    θ = Vector{Float64}(undef, model.D)
    nbad = 0
    keep = trues(N)
    for j in 1:N
        for d in 1:model.D
            θ[d] = cols[d][j]
        end
        θt = try
            model.link(θ)
        catch err
            err isa DomainError || rethrow()
            nothing
        end
        if !isnothing(θt) && all(isfinite, θt)
            X[:, j] .= θt
        else
            keep[j] = false
            nbad += 1
        end
    end
    nbad > 0 && @warn "linked_samples: dropped $nbad of $N draws outside the support of this model's priors"
    return X[:, keep]
end

"""
    mixture_reference_from_chains(model, chains...; split=nothing, weights=nothing,
                                  scale=1.0, first_round=1, min_draws=100)

Build a [`MixtureGaussianReference`](@ref) for `model` from earlier posterior draws:
one diagonal Gaussian per chain (moment-matched in linked space), or two per chain
when `split = (param, cut)` is given (draws with `param < cut` and `param ≥ cut`;
`param` a chain column name such as `:M`). Components with fewer than `min_draws`
draws are skipped with a warning. `scale` multiplies every standard deviation
(> 1 gives heavier coverage of each mode). `weights` as in the constructor
(default equal; it has one entry per component actually built).

The chains can come from runs with other priors (e.g. restricted mass ranges or the
θ-uniform phase prior): they only need to sit on the modes.

```julia
low  = Octofitter.loadchain("family_low_chain.fits")
high = Octofitter.loadchain("family_high_chain.fits")
ref  = mixture_reference_from_chains(model, low, high)
chain, pt = octofit_pigeons(model; n_rounds=17, n_chains=192, n_chains_variational=192,
                            variational=ref)
```
"""
function mixture_reference_from_chains(model::LogDensityModel, chains::MCMCChains.Chains...;
                                       split=nothing, weights=nothing, scale::Real=1.0,
                                       first_round::Integer=1, min_draws::Integer=100)
    isempty(chains) && error("give at least one chain")
    scale > 0 || error("scale must be > 0")
    means = Vector{Float64}[]
    stds = Vector{Float64}[]
    labels = String[]
    for (c, chain) in enumerate(chains)
        groups = if isnothing(split)
            [(Colon(), "chain $c")]
        else
            param, cut = split
            v = vec(Array(chain[Symbol(param)]))
            [(findall(<(cut), v), "chain $c, $(param) < $(cut)"),
             (findall(>=(cut), v), "chain $c, $(param) ≥ $(cut)")]
        end
        for (ii, label) in groups
            X = linked_samples(model, chain; ii)
            if size(X, 2) < min_draws
                @warn "mixture_reference_from_chains: skipping component ($label): $(size(X, 2)) draws < min_draws = $min_draws"
                continue
            end
            μ = vec(mean(X; dims=2))
            σ = vec(std(X; dims=2)) .* scale
            # guard against a frozen coordinate (e.g. a fixed parameter)
            σ .= max.(σ, 1e-8)
            push!(means, μ)
            push!(stds, σ)
            push!(labels, "$label ($(size(X, 2)) draws)")
        end
    end
    isempty(means) && error("no component has ≥ min_draws = $min_draws draws")
    @info "Mixture reference: $(length(means)) components" components = labels
    return MixtureGaussianReference(means, stds; weights, first_round)
end
export mixture_reference_from_chains
