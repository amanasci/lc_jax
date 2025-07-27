import jax
import jax.numpy as jnp
from jax import jit, random
from jax.lax import fori_loop
import matplotlib.pyplot as plt
from scipy.stats import lognorm, kstest
import time

## ==============================================================================
# This is a JAX implementation of the iterative algorithm for generating light curves
# with a specified power spectral density (PSD) and probability density function (PDF).
# The algorithm is based on the work of Emmanoulopoulos et al. (2013).
# It generates a Gaussian-distributed light curve with a given underlying PSD,
# then iteratively adjusts the light curve to match a target PDF while preserving the PSD.  
## =============================================================================

# JAX can be sensitive to 64-bit precision, so we enable it using the updated API.
jax.config.update("jax_enable_x64", True)

# ==============================================================================
# 1. CORE ALGORITHM COMPONENTS
# ==============================================================================

def _generate_gaussian_lightcurve(key, N, psd_model_func):
    """
    Generates a Gaussian-distributed light curve with a given underlying PSD.
    This is an implementation of the Timmer & Koenig (1995) method.
    This is the internal implementation before JIT compilation.

    Args:
        key (jax.random.PRNGKey): JAX random key.
        N (int): The number of points in the light curve.
        psd_model_func (callable): A function that takes frequency `f` and returns the PSD power.

    Returns:
        tuple:
            - jnp.ndarray: The generated Gaussian light curve (time series).
            - jnp.ndarray: The Fourier amplitudes of this light curve.
    """
    # Define the Fourier frequencies
    freqs = jnp.fft.rfftfreq(N)
    
    # Evaluate the PSD model at each frequency
    psd_values = psd_model_func(freqs)

    # Generate random Fourier components
    key_real, key_imag = random.split(key)
    # Draw from a normal distribution and scale by the PSD
    random_amps_real = random.normal(key_real, (len(freqs),)) * jnp.sqrt(0.5 * psd_values)
    random_amps_imag = random.normal(key_imag, (len(freqs),)) * jnp.sqrt(0.5 * psd_values)
    
    # Construct the complex Fourier spectrum
    fourier_spectrum = random_amps_real + 1j * random_amps_imag
    
    # Perform the inverse FFT to get the time series
    # The result is scaled to ensure the variance is correct
    lightcurve = jnp.fft.irfft(fourier_spectrum, n=N) * N
    
    # Get the amplitudes for the spectral adjustment step later
    fourier_amplitudes = jnp.abs(fourier_spectrum)
    
    return lightcurve, fourier_amplitudes

# Apply JIT compilation explicitly, marking arguments that are not JAX arrays/tracers as static.
generate_gaussian_lightcurve = jit(_generate_gaussian_lightcurve, static_argnames=['N', 'psd_model_func'])


@jit
def amplitude_adjustment(x_spectrally_adjusted, x_target_pdf):
    """
    Performs the amplitude adjustment step (Step iv of the paper's algorithm).
    It reorders the values of `x_target_pdf` to match the ranking of `x_spectrally_adjusted`.

    Args:
        x_spectrally_adjusted (jnp.ndarray): The light curve after spectral adjustment.
        x_target_pdf (jnp.ndarray): The light curve with the desired PDF.

    Returns:
        jnp.ndarray: The new light curve with the correct PDF but modified PSD.
    """
    # Get the sort indices from the spectrally adjusted light curve
    sort_indices = jnp.argsort(x_spectrally_adjusted)
    
    # Create the new light curve by rearranging the target PDF light curve
    # according to the ranking of the spectrally adjusted one.
    # We need to apply an argsort twice to get the rank.
    ranks = jnp.argsort(sort_indices)
    
    # Sort the target pdf light curve
    x_target_sorted = jnp.sort(x_target_pdf)
    
    # Reorder the sorted target pdf light curve according to the ranks
    x_new = x_target_sorted[ranks]
    
    return x_new

def generate_lightcurve(key, N, psd_model_func, pdf_dist_obj, num_iterations=100):
    """
    Main function to generate a light curve with a specified PSD and PDF.
    This implements the full iterative algorithm from Emmanoulopoulos et al. (2013).

    Args:
        key (jax.random.PRNGKey): JAX random key.
        N (int): Number of points in the light curve.
        psd_model_func (callable): Function defining the target PSD.
        pdf_dist_obj (scipy.stats object): A frozen Scipy stats distribution object for the target PDF.
        num_iterations (int): The number of iterations to perform.

    Returns:
        jnp.ndarray: The final generated light curve.
    """
    # Split the main key for different random processes
    key_psd, key_pdf, key_loop = random.split(key, 3)

    # --- Step i: Generate a Gaussian light curve to get target Fourier amplitudes ---
    _, target_fourier_amplitudes = generate_gaussian_lightcurve(key_psd, N, psd_model_func)

    # --- Step ii: Generate a light curve with the desired PDF ---
    # NOTE: This part uses scipy, so it runs on the CPU and is not JIT-compiled with the rest.
    # For a pure JAX implementation, you would need a JAX-compatible way to sample
    # from your target distribution.
    x_target_pdf = jnp.array(pdf_dist_obj.rvs(size=N, random_state=int(key_pdf[0])))
    
    # The initial light curve for the iteration is the one with the correct PDF
    x_sim = x_target_pdf

    # --- Iteration Loop (Steps iii, iv, v) ---
    @jit
    def iteration_step(i, x_sim_current):
        # Calculate the FFT of the current simulated light curve
        sim_fft = jnp.fft.rfft(x_sim_current)
        
        # --- Step iii: Spectral Adjustment ---
        # Keep the phases of the current simulation, but enforce the target amplitudes
        sim_phases = jnp.angle(sim_fft)
        adjusted_fft = target_fourier_amplitudes * jnp.exp(1j * sim_phases)
        
        # Inverse FFT to get the spectrally adjusted light curve
        x_spectrally_adjusted = jnp.fft.irfft(adjusted_fft, n=N)
        
        # --- Step iv: Amplitude Adjustment ---
        x_next = amplitude_adjustment(x_spectrally_adjusted, x_target_pdf)
        
        return x_next

    # Run the iterative loop
    final_lightcurve = fori_loop(0, num_iterations, iteration_step, x_sim)

    return final_lightcurve

# ==============================================================================
# 2. EXAMPLE USAGE
# ==============================================================================
if __name__ == '__main__':
    # --- Simulation Parameters ---
    N_POINTS = 2**16  # Number of points in the light curve (power of 2 is good for FFT)
    N_ITERATIONS = 400 # Number of iterations for convergence
    
    # Create a JAX random key
    main_key = random.PRNGKey(int(time.time()))

    # --- Define the Target PSD Model ---
    # A bending power-law model, common in astrophysics
    def bending_power_law_psd(f):
        # Parameters for the model
        A = 1.0       # Normalization
        f_bend = 0.01 # Bend frequency
        alpha_low = 2.0 # Slope at low frequencies
        alpha_high = 2.5 # Slope at high frequencies
        
        # Avoid division by zero at f=0
        f = jnp.maximum(f, 1e-9)
        
        power = A * f**(-alpha_low) / (1 + (f / f_bend)**(alpha_high - alpha_low))
        return power

    # --- Define the Target PDF Model ---
    # A log-normal distribution, which is non-Gaussian and positively skewed.
    # We use a "frozen" scipy.stats object.
    s = 0.954  # Shape parameter for log-normal
    loc = 0    # Location
    scale = 1  # Scale
    lognorm_dist = lognorm(s=s, loc=loc, scale=scale)

    # --- Generate the Light Curve ---
    print(f"Generating light curve with {N_POINTS} points and {N_ITERATIONS} iterations...")
    start_time = time.time()
    
    final_lc = generate_lightcurve(
        key=main_key,
        N=N_POINTS,
        psd_model_func=bending_power_law_psd,
        pdf_dist_obj=lognorm_dist,
        num_iterations=N_ITERATIONS
    )
    # Block until the JAX computation is complete to get accurate timing
    final_lc.block_until_ready()
    end_time = time.time()
    print(f"Generation complete in {end_time - start_time:.2f} seconds.")

    # ==========================================================================
    # 3. VISUALIZE AND VALIDATE THE RESULTS
    # ==========================================================================
    
    # --- Plot the final light curve ---
    plt.figure(figsize=(15, 12))
    
    # 1. Time series plot
    ax1 = plt.subplot(3, 1, 1)
    time_axis = jnp.arange(N_POINTS)
    ax1.plot(time_axis, final_lc, lw=0.7, color='teal')
    ax1.set_title(f'Generated Light Curve (N={N_POINTS}, Iterations={N_ITERATIONS})')
    ax1.set_xlabel('Time (arbitrary units)')
    ax1.set_ylabel('Flux (arbitrary units)')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Validate the PDF ---
    ax2 = plt.subplot(3, 2, 3)
    # Plot histogram of the generated data
    ax2.hist(final_lc, bins=100, density=True, alpha=0.7, label='Generated LC PDF', color='cornflowerblue')
    # Plot the target PDF
    x_pdf = jnp.linspace(lognorm_dist.ppf(0.001), lognorm_dist.ppf(0.999), 200)
    ax2.plot(x_pdf, lognorm_dist.pdf(x_pdf), 'r-', lw=2, label='Target Log-Normal PDF')
    ax2.set_title('PDF Validation')
    ax2.set_xlabel('Flux')
    ax2.set_ylabel('Probability Density')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Perform a Kolmogorov-Smirnov test to quantify the match
    ks_statistic, p_value = kstest(final_lc, lognorm_dist.cdf)
    print(f"\nKolmogorov-Smirnov Test for PDF match:")
    print(f"  - KS Statistic: {ks_statistic:.4f}")
    print(f"  - p-value: {p_value:.4f}")
    if p_value > 0.05:
        print("  - Result: The generated distribution is consistent with the target PDF (p > 0.05).")
    else:
        print("  - Result: The generated distribution is NOT consistent with the target PDF (p <= 0.05).")


    # --- Validate the PSD ---
    ax3 = plt.subplot(3, 2, 4)
    # Calculate the periodogram (the actual PSD of the generated light curve)
    freqs = jnp.fft.rfftfreq(N_POINTS)
    periodogram = jnp.abs(jnp.fft.rfft(final_lc))**2
    
    # Plot the periodogram and the target PSD model
    ax3.loglog(freqs[1:], periodogram[1:], label='Generated LC Periodogram', alpha=0.5, color='gray', lw=0.5)
    ax3.loglog(freqs[1:], bending_power_law_psd(freqs[1:]) * (0.5 * N_POINTS**2), 'r-', label='Target PSD Model', lw=2)
    ax3.set_title('PSD Validation')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Power')
    ax3.legend()
    ax3.grid(True, which='both', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig("/run/media/asci/Data/E/Projects2/lc_jax/plots/final_light_curve_validation.png")
    plt.show()
