# LC JAX

This is a JAX implementation of the iterative algorithm for generating light curves with a specified power spectral density (PSD) and probability density function (PDF).

The main algorithm is based on the work of Emmanoulopoulos et al. (2013). It generates a Gaussian-distributed light curve with a given underlying PSD, then iteratively adjusts the light curve to match a target PDF while preserving the PSD.

## Example

A sample of the generate light curve is shown below. The light curve is generated with a specified PSD and PDF, and the iterative process ensures that the final light curve matches the target PDF while maintaining the desired PSD.

![Light Curve Example](https://raw.githubusercontent.com/asci/lc_jax/main/plots/final_light_curve_validation.png)
