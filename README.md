This repository implements Path Shadowing Monte-Carlo [1], which can be used for volatility prediction and option pricing.

This methods averages future quantities over generated price paths whose past history matches, or `shadows', the actual (observed) history.

<p align="center">
<figure>
    <img src="./illustration/anim_shadowing.gif" alt="animated">
    <!-- <img src="https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif" alt="animated"> -->
<figcaption> Simple shadowing example. Red: observed S&P daily log-returns, 

grey (shadow): visualization of collected shadowing paths from our model. </figcaption>
</figure>
</p>

# Prediction / Option pricing

The class PathShadowing from `path_shadowing.py` implements a multi-processed scan of a generated dataset for shadowing paths.

Notebook `tutorial.ipynb` shows how to use it. 



# Generation 

The paper uses the Scattering Spectra [2] to generate the dataset of time-series.

Such generative model is implemented by the package **scatspectra**:

```bash
pip install git+https://github.com/RudyMorel/scattering_spectra
```



[1] "Path Shadowing Monte-Carlo"

Rudy Morel et al. - https://arxiv.org/abs/2308.01486

[2] "Scale Dependencies and Self-Similar Models with Wavelet Scattering Spectra"

Rudy Morel et al. - https://arxiv.org/abs/2204.10177

