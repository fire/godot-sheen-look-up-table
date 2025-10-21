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
from scipy.optimize import minimize

def main():
    # Load data
    with open("thirdparty/sheen_lut_data.txt", "r") as f:
        content = f.read()

    # Parse the list
    data_str = content.strip()[1:-1]  # Remove [ and ]
    data = [float(val.strip()) for val in data_str.split(",")]

    # Reshape to 128x128
    lut = np.array(data).reshape(128, 128)
    print(f"Original LUT range: min={np.min(lut):.4f}, max={np.max(lut):.4f}")

    # Raw array indices
    r = np.arange(128)  # roughness indices 0-127
    c = np.arange(128)  # cos_theta indices 0-127

    # Fit 2D polynomial of degree 4 for improved fit under 0.001 MSE
    deg = 4
    terms = []
    for i in range(deg + 1):
        for j in range(deg + 1 - i):
            terms.append((i, j))

    n_terms = len(terms)
    R_, C_ = np.meshgrid(r, c, indexing='ij')

    X = np.zeros((128 * 128, n_terms))
    k = 0
    for i, j in terms:
        X[:, k] = R_.ravel() ** i * C_.ravel() ** j
        k += 1

    # Initial least squares fit as starting point
    initial_coeffs, _, _, _ = np.linalg.lstsq(X, lut.ravel(), rcond=None)

    def negative_ssim(coeffs):
        approx = np.dot(X, coeffs).reshape(128, 128)
        return -ssim(lut, np.clip(approx, 0, None), data_range=np.max(lut))

    # Optimize coefficients for maximum SSIM
    res = minimize(negative_ssim, initial_coeffs, method='BFGS')
    coeffs = res.x

    # Symbolic variables
    x, y = sympy.symbols('r cos_theta')  # assuming r is roughness, cos_theta is something

    # Build the expression
    expr = sum(coeffs[k] * x**i * y**j for k, (i, j) in enumerate(terms))

    # Optional: check the fit quality
    lut_approx = np.dot(X, coeffs).reshape(128, 128)
    print(f"Approximated LUT range: min={np.min(lut_approx):.4f}, max={np.max(lut_approx):.4f}")
    ssim_val = ssim(lut, np.clip(lut_approx, 0, None), data_range=np.max(lut))
    print(f"SSIM optimization success: {res.success}")
    if not res.success:
        print(f"Optimization message: {res.message}")
    print("Approximated analytical expression for the sheen LUT:")
    print(sympy.simplify(expr))
    print(f"Structural Similarity Index (SSIM): {ssim_val:.4f}")

    # Clip negative values in approximation for visualization (as sheen values can't be negative)
    lut_display = lut
    lut_approx_display = np.clip(lut_approx, 0, None)

    # Use percentile scaling for better contrast and visibility (shared across both images)
    vmin = np.percentile(lut_display, 5)
    vmax = np.percentile(lut_display, 95)

    print(f"Display range (5th to 95th percentile): vmin={vmin:.4f}, vmax={vmax:.4f}")

    plt.figure()
    plt.imshow(lut_display, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.savefig('original_lut.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.imshow(lut_approx_display, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.savefig('approximated_lut.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Images saved as original_lut.png and approximated_lut.png")

if __name__ == "__main__":
    main()
