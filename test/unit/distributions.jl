@testset "Special Prior Distributions" begin
    @testset "Sine Distribution" begin
        sine = Sine()
        @test Distributions.minimum(sine) ≈ 0.0 + eps()
        @test Distributions.maximum(sine) ≈ π - eps()
        @test Distributions.insupport(sine, π/2)
        @test !Distributions.insupport(sine, -0.1)
        @test !Distributions.insupport(sine, π + 0.1)

        # Test PDF shape - peak should be at π/2
        @test Distributions.pdf(sine, π/2) > Distributions.pdf(sine, 0.1)
        @test Distributions.pdf(sine, π/2) > Distributions.pdf(sine, π-0.1)
    end

    @testset "UniformCircular" begin
        uc = UniformCircular()
        @test uc.domain ≈ 2π

        uc_custom = UniformCircular(1.0)
        @test uc_custom.domain ≈ 1.0

        # Test in model context
        @test_nowarn Planet(
            name="b",
            basis=Visual{KepOrbit},
            observations=[],
            variables=@variables begin
                a ~ LogUniform(0.1, 100)
                e ~ Uniform(0, 0.99)
                i ~ Sine()
                ω ~ UniformCircular()
                Ω ~ UniformCircular(π)  # Custom domain
                θ ~ UniformCircular()
            end
        )
    end

    @testset "KDE Priors" begin
        samples = randn(1000) .+ 10  # Normal(10,1) samples
        kde = Octofitter.KDEDist(samples)

        @test kde isa Octofitter.KDEDist
        @test Distributions.minimum(kde) ≈ minimum(samples)
        @test Distributions.maximum(kde) ≈ maximum(samples)
        @test Distributions.insupport(kde, 10.0)
        @test !Distributions.insupport(kde, minimum(samples) - 1)

        # Test in model context
        @test_nowarn Planet(
            name="b",
            basis=Visual{KepOrbit},
            observations=[],
            variables=@variables begin
                a ~ kde
                e ~ Uniform(0, 0.99)
                i ~ Sine()
                ω ~ UniformCircular()
                Ω ~ UniformCircular()
                θ ~ UniformCircular()
                tp = θ_at_epoch_to_tperi(θ, 50000; M=system.M, e, a, i, ω, Ω)
            end
        )
    end
end

@testset "Observable-based Priors" begin
    astrom_data = PlanetRelAstromLikelihood(
        Table(epoch = [50000, 50100], ra = [100.0, 110.0], dec = [50.0, 55.0], σ_ra = [1.0, 1.0], σ_dec = [1.0, 1.0]),
        name="obs_prior_test"
    )

    obs_prior = ObsPriorAstromONeil2019(astrom_data)
    @test obs_prior isa ObsPriorAstromONeil2019
    @test obs_prior.wrapped_like === astrom_data

    # Test in model context with period prior
    @test_nowarn Planet(
        name="b",
        basis=Visual{KepOrbit},
        observations=[astrom_data, obs_prior],
        variables=@variables begin
            e ~ Uniform(0.0, 0.5)
            i ~ Sine()
            ω ~ UniformCircular()
            Ω ~ UniformCircular()
            P ~ LogUniform(0.1, 150)
            a = ∛(system.M * P^2)
            θ ~ UniformCircular()
            tp = θ_at_epoch_to_tperi(θ, 50000; M=system.M, e, a, i, ω, Ω)
        end
    )

    # Test subsetting
    subset = Octofitter.likeobj_from_epoch_subset(obs_prior, 1:1)
    @test subset isa ObsPriorAstromONeil2019
end

@testset "Observable-based prior is |J| to the first power" begin
    # O'Neil et al. 2019 eq. 36a.  The same model with and without the wrapper,
    # evaluated at the same parameters, must differ by exactly log|J| (this was
    # 2log|J| before 2026-10).  |J| is computed independently from eq. 33.
    epoch = 55927.0
    astrom = PlanetRelAstromObs(
        Table(epoch=[epoch], ra=[-260.0], dec=[-51.0], σ_ra=[0.5], σ_dec=[0.5], cor=[0.0]);
        name="pos")
    function build(obs)
        b = Planet(name="A", basis=Visual{KepOrbit}, observations=[obs],
            variables=@variables begin
                M = system.M
                P ~ Uniform(10, 2_000_000)
                a = cbrt(M * P^2)
                e ~ Uniform(0.0, 0.99)
                i ~ Sine()
                ω ~ UniformCircular()
                Ω ~ UniformCircular()
                θ ~ UniformCircular()
                tp = θ_at_epoch_to_tperi(θ, $epoch; a, e, i, ω, Ω, M)
            end)
        sys = System(name="oneil_exponent", companions=[b], observations=[],
            variables=@variables begin
                plx ~ truncated(Normal(0.1842, 0.0017), lower=0)
                M ~ Uniform(100, 100000)
            end)
        Octofitter.LogDensityModel(sys; verbosity=0)
    end
    m_plain = build(astrom)
    m_prior = build(ObsPriorAstromONeil2019(astrom))
    rng = Octofitter.Random.Xoshiro(3)   # Random is loaded by Octofitter, not by runtests.jl
    for _ in 1:4
        θ  = m_plain.sample_priors(rng)
        θt = m_plain.link(θ)
        term = m_prior.ℓπcallback(θt) - m_plain.ℓπcallback(θt)
        nt = m_plain.arr2nt(θ)
        pl = nt.planets.A
        orb = Visual{KepOrbit}(; a=pl.a, e=pl.e, i=pl.i, ω=pl.ω, Ω=pl.Ω, tp=pl.tp, M=nt.M, plx=nt.plx)
        sol = orbitsolve(orb, epoch)
        e, m, E = pl.e, meananom(sol), eccanom(sol)
        jac = abs(3m*(e+cos(E)) + 2*(-2+e^2+e*cos(E))*sin(E)) * cbrt(period(orb)/365.25) / sqrt(1-e^2)
        @test term ≈ log(jac) rtol=1e-6
    end
end

@testset "θ phase prior: uniform in time (PlanetThetaPhasePriorObs)" begin
    # With θ ~ UniformCircular() and tp = θ_at_epoch_to_tperi(θ, epoch; ...), the
    # implied prior density in mean anomaly is |dθ/dMA|/2π.  PlanetThetaPhasePriorObs
    # adds −log|dθ/dMA| = log ρ² − log|cos i| − 2 log a − ½ log(1−e²) so that the phase
    # prior is uniform in time (O'Neil et al. 2019 eq. 30).
    epoch = 55927.0
    Mc, plx0 = 5e4, 0.1842
    astrom = PlanetRelAstromObs(
        Table(epoch=[epoch], ra=[-260.0], dec=[-51.0], σ_ra=[0.5], σ_dec=[0.5], cor=[0.0]);
        name="pos")
    function build_phase(obs)
        b = Planet(name="A", basis=Visual{KepOrbit}, observations=obs,
            variables=@variables begin
                M = system.M
                P ~ Uniform(10, 2_000_000)
                a = cbrt(M * P^2)
                e ~ Uniform(0.0, 0.99)
                i ~ Sine()
                ω ~ UniformCircular()
                Ω ~ UniformCircular()
                θ ~ UniformCircular()
                tp = θ_at_epoch_to_tperi(θ, $epoch; a, e, i, ω, Ω, M)
            end)
        sys = System(name="theta_phase", companions=[b], observations=[],
            variables=@variables begin
                plx ~ truncated(Normal(0.1842, 0.0017), lower=0)
                M ~ Uniform(100, 100000)
            end)
        Octofitter.LogDensityModel(sys; verbosity=0)
    end
    phase_obs = PlanetThetaPhasePriorObs(epoch; name="phase")

    @testset "construction and subsetting" begin
        @test phase_obs isa Octofitter.AbstractObs
        @test phase_obs.table.epoch == [epoch]
        sub = Octofitter.likeobj_from_epoch_subset(phase_obs, 1:1)
        @test sub isa PlanetThetaPhasePriorObs
        @test sub.table.epoch == [epoch]
    end

    @testset "exact term in the log density" begin
        m_plain = build_phase([astrom])
        m_phase = build_phase([astrom, phase_obs])
        rng = Octofitter.Random.Xoshiro(7)
        for _ in 1:6
            θ  = m_plain.sample_priors(rng)
            θt = m_plain.link(θ)
            term = m_phase.ℓπcallback(θt) - m_plain.ℓπcallback(θt)
            nt = m_plain.arr2nt(θ)
            pl = nt.planets.A
            orb = Visual{KepOrbit}(; a=pl.a, e=pl.e, i=pl.i, ω=pl.ω, Ω=pl.Ω, tp=pl.tp, M=nt.M, plx=nt.plx)
            sol = orbitsolve(orb, epoch)
            ρ2 = posx(sol)^2 + posy(sol)^2
            expected = log(ρ2) - log(abs(cos(pl.i))) - 2log(pl.a) - log(1 - pl.e^2)/2
            # ℓ at random prior draws is large (far from the astrometry), so the
            # difference carries cancellation error; same tolerance as the O'Neil test.
            @test isapprox(term, expected; rtol=1e-6, atol=1e-6)
        end
    end

    @testset "dθ/dMA formula vs finite differences" begin
        rng = Octofitter.Random.Xoshiro(11)
        for _ in 1:25
            a  = 5e3 + 2e4*rand(rng)
            e  = 0.95*rand(rng)
            i  = rand(rng) < 0.5 ? 0.1 + 1.3*rand(rng) : π - 0.1 - 1.3*rand(rng)   # |cos i| ≳ 0.17
            ω, Ω, MA0 = 2π*rand(rng), 2π*rand(rng), 2π*rand(rng)
            orb = Visual{KepOrbit}(; a, e, i, ω, Ω, tp=0.0, M=Mc, plx=plx0)
            θof(MA) = posangle(orbitsolve_meananom(orb, MA))
            h = 1e-6
            dθ = rem2pi(θof(MA0 + h) - θof(MA0 - h), RoundNearest) / (2h)
            sol = orbitsolve_meananom(orb, MA0)
            ρ2 = posx(sol)^2 + posy(sol)^2
            formula = abs(cos(i)) * a^2 * sqrt(1 - e^2) / ρ2
            @test abs(dθ) ≈ formula rtol=1e-4
            @test Octofitter._theta_phase_logjac(ρ2, cos(i), a, e) ≈ -log(formula) rtol=1e-10
        end
    end

    @testset "weighted phase is uniform in time; a, e, i marginals unchanged" begin
        # Draw from the θ-uniform prior, weight by exp(term).  Inclinations are kept
        # away from edge-on: under the prior alone the weights have a log-divergent
        # variance as |cos i| → 0 (their mean is exactly 1), which would make this
        # Monte Carlo check unstable without changing its conclusion.
        rng = Octofitter.Random.Xoshiro(2026)
        N = 40_000
        MA = zeros(N); w = zeros(N); es = zeros(N); cis = zeros(N); as = zeros(N)
        for k in 1:N
            a = 5e3 + 2e4*rand(rng)
            e = 0.9*rand(rng)
            i = 0.3 + 0.9*rand(rng)
            ω, Ω = 2π*rand(rng), 2π*rand(rng)
            θ = 2π*rand(rng) - π
            tp = θ_at_epoch_to_tperi(θ, epoch; a, e, i, ω, Ω, M=Mc)
            orb = Visual{KepOrbit}(; a, e, i, ω, Ω, tp, M=Mc, plx=plx0)
            sol = orbitsolve(orb, epoch)
            MA[k] = mod(meananom(sol), 2π)
            w[k] = exp(Octofitter._theta_phase_logjac(posx(sol)^2 + posy(sol)^2, cos(i), a, e))
            es[k], cis[k], as[k] = e, cos(i), a
        end
        W = w ./ sum(w)
        Neff = 1 / sum(W .^ 2)
        @test Neff > N / 20

        nb = 8
        bins = clamp.(floor.(Int, MA ./ (2π/nb)) .+ 1, 1, nb)
        p_w = [sum(W[bins .== b]) for b in 1:nb]
        σp = sqrt((1/nb) * (1 - 1/nb) / Neff)
        @test all(abs.(p_w .- 1/nb) .< 5σp)

        # Unweighted (θ-uniform) phases pile up at pericentre for eccentric orbits
        ecc = es .> 0.7
        near_peri = (MA .< π/8) .| (MA .> 2π - π/8)
        @test sum(near_peri .& ecc) / sum(ecc) > 2 * (1/8)

        # The Jacobian integrates to 2π over phase: orbit-shape marginals unchanged
        for x in (es, cis, as)
            μu = sum(x) / N
            μw = sum(W .* x)
            σμ = sqrt(sum(W .* (x .- μw).^2) / Neff)
            @test abs(μw - μu) < 5σμ
        end
    end

    @testset "degenerate geometry returns -Inf, never NaN or +Inf" begin
        f = Octofitter._theta_phase_logjac
        @test f(0.0, 0.5, 1e4, 0.3) == -Inf
        @test f(1e8, 0.0, 1e4, 0.3) == -Inf
        @test f(1e8, 1e-13, 1e4, 0.3) == -Inf
        @test f(1e8, 0.5, 1e4, 1.0) == -Inf
        @test isfinite(f(1e8, -0.5, 1e4, 0.3))          # cos i < 0 (retrograde) is fine
        @test f(1e8, -0.5, 1e4, 0.3) == f(1e8, 0.5, 1e4, 0.3)
    end

    @testset "AD gradient finite and consistent" begin
        # Phase term only (no astrometry), so the finite differences are not swamped
        # by a huge likelihood at random prior draws.
        m_phase = build_phase([phase_obs])
        rng = Octofitter.Random.Xoshiro(5)
        for _ in 1:3
            θt = m_phase.link(m_phase.sample_priors(rng))
            ℓ, g = m_phase.∇ℓπcallback(θt)
            g = copy(g)
            @test isfinite(ℓ)
            @test all(isfinite, g)
            # central finite differences on a few coordinates
            for j in (1, 2, 3, length(θt))
                δ = zeros(length(θt)); δ[j] = 1e-6
                fd = (m_phase.ℓπcallback(θt .+ δ) - m_phase.ℓπcallback(θt .- δ)) / 2e-6
                @test isapprox(g[j], fd; rtol=1e-3, atol=1e-5)
            end
        end
    end

    @testset "LogUniform mass prior links and differentiates" begin
        b = Planet(name="A", basis=Visual{KepOrbit}, observations=[astrom, phase_obs],
            variables=@variables begin
                M = system.M
                P ~ Uniform(10, 2_000_000)
                a = cbrt(M * P^2)
                e ~ Uniform(0.0, 0.99)
                i ~ Sine()
                ω ~ UniformCircular()
                Ω ~ UniformCircular()
                θ ~ UniformCircular()
                tp = θ_at_epoch_to_tperi(θ, $epoch; a, e, i, ω, Ω, M)
            end)
        sys = System(name="loguniform_M", companions=[b], observations=[],
            variables=@variables begin
                plx ~ truncated(Normal(0.1842, 0.0017), lower=0)
                M ~ LogUniform(100, 100000)
            end)
        m = Octofitter.LogDensityModel(sys; verbosity=0)
        rng = Octofitter.Random.Xoshiro(9)
        for _ in 1:3
            θ = m.sample_priors(rng)
            @test 100 <= m.arr2nt(θ).M <= 100000
            @test collect(m.invlink(m.link(θ))) ≈ collect(θ)   # sample_priors returns a tuple
            ℓ, g = m.∇ℓπcallback(m.link(θ))
            @test isfinite(ℓ) && all(isfinite, g)
        end
    end
end
