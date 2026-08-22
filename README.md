# Octofitter.jl

> **This is a development fork** of [Octofitter.jl](https://github.com/sefffal/Octofitter.jl)
> (v8.1.2), extended for fitting the orbits of fast-moving stars around an
> intermediate-mass black hole candidate in the globular cluster ω Centauri.
> See [Fork modifications](#fork-modifications) below. All other documentation
> refers to upstream Octofitter.

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://sefffal.github.io/Octofitter.jl/dev/)

Octofitter is a Julia package for performing Bayesian inference 
against a wide variety of exoplanet / binary star data.
You can also use Octofitter from Python using [octofitterpy](https://github.com/sefffal/octofitterpy).

**Read the tutorials and documentation [here](https://sefffal.github.io/Octofitter.jl/)**.

![](docs/src/assets/gallery.png)


### Read the paper
In addition to these documentation and tutorial pages, you can read the paper published in the [Astronomical Journal](https://dx.doi.org/10.3847/1538-3881/acf5cc) (open-access).

## Fork modifications

This fork adds four features to upstream Octofitter, all backwards compatible:
if the corresponding variables are not declared, the code reduces to stock
Octofitter behaviour.

| Feature | File | Summary |
|---|---|---|
| Free central mass position | `src/likelihoods/relative-astrometry.jl` | System-level `offsetx`/`offsety` shift the assumed central-mass position (RA/Dec, in the units of the astrometry table) |
| `PlanetPMObs` | `src/likelihoods/proper-motion.jl` | Direct 2D proper-motion likelihood |
| `PlanetAccelObs` | `src/likelihoods/acceleration.jl` | Direct 2D acceleration likelihood |
| `PlanetZPriorObs` | `src/likelihoods/z-prior.jl` | Prior on the line-of-sight offset from the central mass |

`offsetx`/`offsety` are read from `θ_system` rather than `θ_obs`, because the
central mass position is physically shared by all orbiting objects and must not
vary per observation. Declare them in the `System`-level `@variables` block.

### Caveat: the position offsets apply only to RA/Dec astrometry

`PlanetRelAstromObs` accepts astrometry either as `(ra, dec)` or as
`(sep, pa)`. **The `offsetx`/`offsety` correction is applied only in the
`(ra, dec)` branch.** If a table supplies separation and position angle
instead, both `ln_like` and `generate_from_params` compute residuals with no
offset term, and the free central mass position silently has no effect —
the sampler will explore `offsetx`/`offsety` while the likelihood remains
flat in those parameters.

Supply astrometry as `(ra, dec)` whenever a free central mass position is
being fitted. Adding the offsets to the `(sep, pa)` branch is not simply a
matter of subtracting them, since an offset in the tangent plane is not a
separable shift in `(sep, pa)`; the observed values would need to be
converted to `(ra, dec)`, offset, and converted back.

## Attribution
* If you use Octofitter in your work, please cite [Thompson et al](https://dx.doi.org/10.3847/1538-3881/acf5cc)
* If you use any of the plots, please consider citing the plotting library Makie.jl [Danisch & Krumbiegel, (2021).](https://doi.org/10.21105/joss.03349)
* If you use Gaia parallaxes in your work, please cite Gaia DR3 [Gaia Collaboration et al. 2023](https://ui.adsabs.harvard.edu/abs/2023A&A...674A...1G)
* If you use Hipparcos-GAIA proper motion anomaly, please cite [Brandt 2021](https://ui.adsabs.harvard.edu/abs/2021ApJS..254...42B)
* If you use example data in one of the tutorials, please cite the sources listed [Brandt 2021](https://ui.adsabs.harvard.edu/abs/2021ApJS..254...42B)
* If you use one of the included functions for automatically retreiving data from a public dataset, eg HARPS RVBank, please cite the source as appropriate.
* If you adopt the O'Neil et al. 2019 observable based priors, please cite [O'Neil et al. 2019](https://ui.adsabs.harvard.edu/abs/2019AJ....158....4O).
* Please also consider citing the HMC sampler backend, [Xu et al 2020](http://proceedings.mlr.press/v118/xu20a.html)
* If you use the corner plot functionality, please cite:
```
@misc{Thompson2023,
  author = {William Thompson},
  title = {{PairPlots.jl} Beautiful and flexible visualizations of high dimensional data},
  year = {2023},
  howpublished = {\url{https://sefffal.github.io/PairPlots.jl/dev}},
}
```


## Ready?


For instructions, see the documentation page:

[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://sefffal.github.io/Octofitter.jl/dev/)
