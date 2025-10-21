# Copyright (c) 2025-present K. S. Ernest (iFire) Lee
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import sympy
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def main():
    # Load data
    with open("thirdparty/sheen_lut_data.txt", "r") as f:
        content = f.read()

    # Parse the list
    data_str = content.strip()[1:-1]  # Remove [ and ]
    data = [float(val.strip()) for val in data_str.split(",")]

    # Reshape to 128x128
    lut = np.array(data).reshape(128, 128)
    vmax = np.max(lut)
    print(f"Original LUT range: min={np.min(lut):.4f}, max={vmax:.4f}")
    print(f"LUT max value for denormalization: {vmax}")

    # Normalize LUT to compress range to 0-1 for better fit
    norm_lut = lut / vmax

    # Normalized inputs (0-1) for better numerical fit and higher SSIM
    # In shader: compute norm_roughness = roughness_index / 127.0, norm_cos_theta = cos_theta_index / 127.0
    # Then: output = polynomial(norm_roughness, norm_cos_theta) * vmax
    r = np.linspace(0, 1, 128)  # roughness normalized 0-1
    c = np.linspace(0, 1, 128)  # cos_theta normalized 0-1

    # Fit triangular 2D polynomial of degree 16 for higher precision ((17+1)*(18)/2 = 153 terms)
    deg = 16
    terms = [(i, j) for i in range(deg + 1) for j in range(deg - i + 1)]

    n_terms = len(terms)
    R_, C_ = np.meshgrid(r, c, indexing='ij')

    X = np.zeros((128 * 128, n_terms))
    k = 0
    for i, j in terms:
        X[:, k] = R_.ravel() ** i * C_.ravel() ** j
        k += 1

    coeffs = np.linalg.lstsq(X, norm_lut.ravel(), rcond=None)[0]
    # Approximate coefficients to float32 precision
    coeffs = np.round(coeffs, 7)

    # Symbolic variables
    x, y = sympy.symbols('roughness cos_theta')

    # Build the expression
    expr = sum(coeffs[k] * x**i * y**j for k, (i, j) in enumerate(terms))

    # Check fit quality
    norm_approx = np.dot(X, coeffs).reshape(128, 128)
    lut_approx = norm_approx * vmax
    mse = np.mean((lut - lut_approx)**2)
    ssim_val = ssim(lut, np.clip(lut_approx, 0, None), data_range=vmax)
    print("Approximated analytical expression for the sheen LUT (normalized inputs 0-1):")
    # Note: In shader, compute norm_roughness = roughness_index / 127.0, norm_cos_theta = cos_theta_index / 127.0
    # Then: sheen_value = polynomial_output * vmax
    print(sympy.simplify(expr))
    print(f"Mean Squared Error: {mse:.8f}")
    print(f"Structural Similarity Index (SSIM): {ssim_val:.4f}")

    # Save images
    vmin = 0.0
    vmax = np.max(lut)

    plt.figure()
    plt.imshow(lut, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    plt.title('Original LUT')
    plt.axis('off')
    plt.savefig('original_lut.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.imshow(lut_approx, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    plt.title('Approximated LUT')
    plt.axis('off')
    plt.savefig('approximated_lut.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create comparison image
    diff = np.abs(lut - lut_approx)
    max_diff = np.max(diff)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(lut, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[0].set_title('Original LUT')
    axes[0].axis('off')

    axes[1].imshow(lut_approx, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[1].set_title('Approximated LUT')
    axes[1].axis('off')

    axes[2].imshow(diff, cmap='gray', origin='lower')
    axes[2].set_title(f'Difference (max: {max_diff:.4e})')
    axes[2].axis('off')

    plt.savefig('comparison_lut.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
