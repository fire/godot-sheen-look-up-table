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
    print(f"Original LUT range: min={np.min(lut):.4f}, max={np.max(lut):.4f}")

    # Normalized coordinates (0-1 for stability in fitting)
    r = np.linspace(0, 1, 128)  # roughness 0-1
    c = np.linspace(0, 1, 128)  # cos_theta 0-1

    # Fit triangular 2D polynomial of higher degree 10 for more precision in low areas (e.g., top left corner)
    deg = 10
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

    # Least squares fit (minimizes MSE)
    coeffs, residuals, rank, s = np.linalg.lstsq(X, lut.ravel(), rcond=None)

    # Symbolic variables
    x, y = sympy.symbols('roughness cos_theta')  # normalized 0-1

    # Build the expression
    expr = sum(coeffs[k] * x**i * y**j for k, (i, j) in enumerate(terms))

    # Optional: check the fit quality
    lut_approx = np.dot(X, coeffs).reshape(128, 128)
    mse = np.mean((lut - lut_approx)**2)
    ssim_val = ssim(lut, np.clip(lut_approx, 0, None), data_range=np.max(lut))
    print("Approximated analytical expression for the sheen LUT (normalize inputs to 0-1):")
    print(sympy.simplify(expr))
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"Structural Similarity Index (SSIM): {ssim_val:.4f}")

    # Save individual images
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

    # Create comparison image to highlight differences
    diff = np.abs(lut - lut_approx)
    max_diff = np.max(diff)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(lut, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[0].set_title('Original LUT')
    axes[0].axis('off')
    axes[1].imshow(lut_approx, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    axes[1].set_title('Approximated LUT')
    axes[1].axis('off')
    im = axes[2].imshow(diff, cmap='hot', origin='lower', vmin=0, vmax=max_diff)
    axes[2].set_title('Absolute Difference (|Original - Approx|)')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])
    plt.tight_layout()
    plt.savefig('comparison_lut.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Images saved: original_lut.png, approximated_lut.png, comparison_lut.png (side-by-side with difference)")

if __name__ == "__main__":
    main()
