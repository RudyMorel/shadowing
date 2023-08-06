This repository implements Path Shadowing Monte-Carlo [1], which can be used for volatility prediction or option pricing.

This methods averages future quantities over generated price paths whose past history matches, or `shadows', the actual (observed) history.



# Generation 

Use the function generate() from `frontend.py` to generate the dataset of trajectories that is scanned for shadowing paths.

It uses a Scattering Spectra model introduced in [2].


# Prediction / Option pricing

The class PathShadowing from `path_shadowing.py` implements a multi-processed scan of a generated dataset.




[1] "Path Shadowing Monte-Carlo"

Rudy Morel et al. - https://arxiv.org/abs/2308.01486

[2] "Scale Dependencies and Self-Similar Models with Wavelet Scattering Spectra"

Rudy Morel et al. - https://arxiv.org/abs/2204.10177