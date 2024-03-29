Implements Path Shadowing Monte-Carlo [1], which can be used for volatility prediction and option pricing.

This methods averages future quantities over generated price paths (grey) whose past history matches, or `shadows', the actual observed history (red).

<p align="center">
    <img src="./illustration/anim_shadowing.gif" alt="animated" width="400px"/>
</p>

# Prediction / Option pricing

The class PathShadowing from `path_shadowing.py` implements a scan of a generated dataset for shadowing paths.

Notebook `tutorial.ipynb` shows how to use it. 



# Generation 

The paper uses the Scattering Spectra [2] to generate the dataset of time-series.

[1] "Path Shadowing Monte-Carlo"

Rudy Morel et al. - https://arxiv.org/abs/2308.01486

[2] "Scale Dependencies and Self-Similar Models with Wavelet Scattering Spectra"

Rudy Morel et al. - https://arxiv.org/abs/2204.10177



