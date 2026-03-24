	Hi [@Vincent Henault-Brunet](https://octofitter.slack.com/team/U09D9G6DA9Z), that's great to hear! I'm glad you were able to get it converged on a single node (192 chains is quite a few, so I'm relieved you didn't need to go higher).The current version of octofitter you are using has the ability to marginalize over uncertainty in the rotation and scaling of the coordinate space.  
I'm fully saturated for requested Octofitter modifications for the time being, but I can point you to exactly what part of the code would need to change, and exactly how to change it.

Here are the relevant contents of the relative astrometry likelihood:

```
platescale = hasproperty(θ_obs, :platescale) ? 
getproperty(θ_obs, :platescale) : one(T) 
northangle = hasproperty(θ_obs, :northangle) ? 
getproperty(θ_obs, :northangle) : zero(T) 

# ...

pa_dat = atan(astrom.table.dec[i_epoch], astrom.table.ra[i_epoch]) + northangle sep_dat = hypot(astrom.table.dec[i_epoch], astrom.table.ra[i_epoch]) * platescale 
ra_dat = sep_dat * cos(pa_dat)
dec_dat = sep_dat * sin(pa_dat) 

# calculate residuals 
resid1 = ra_dat - ra_model
resid2 = dec_dat - dec_model
```


We read the `northangle` and `platescale` values from the observation variables from this particular parameter draw.  
Then, we rotate and scale the astrometry data by northangle and platescale respectively.What you could do is add two optional variables, say, `offsetx` and `offsety`  and shift the data by that amount:  

`offsetx = hasproperty(θ_obs, :offsetx) ? getproperty(θ_obs, :offsetx) : zero(T)`
`offsety = hasproperty(θ_obs, :offsety) ? getproperty(θ_obs, :offsety) : zero(T)`
`...`

Then just add those shifts to the data before calculating the residuals  

`resid1 = ra_dat - offsetx - ra_model`
`resid2 = dec_dat -offsety - dec_model`

To modify the Octofitter source code, do  

`using Pkg; Pkg.develop("Octofitter")`

That will clone the package and place it into `~/.julia/dev/Octofitter`The file that needs to be changed is `~/.julia/dev/Octofitter/src/likelihoods/relative-astrometry.jl`
