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
from numpy.polynomial.chebyshev import chebval
from scipy.interpolate import RectBivariateSpline

def fit_bilinear_plane(lut_region, r_norm, c_norm):
    """Fit bilinear plane a + b*r + c*c to region data"""
    r_region = r_norm[:, None]
    c_region = c_norm[None, :]
    X = np.ones((r_region.shape[0] * c_region.shape[1], 3))
    X[:, 1] = r_region.ravel()
    X[:, 2] = c_region.ravel()
    y = lut_region.ravel()
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return coeffs  # [a, b, c]

def evaluate_region(r_eval, c_eval, region):
    """Evaluate bilinear plane for a region"""
    i_min, i_max, j_min, j_max, a, b, c = region
    # Normalize r_eval and c_eval to region center or something? No, for analytical continuity, perhaps not.
    # For simplicity, use global r_norm, c_norm scaled.
    return a + b * r_eval + c * c_eval

def build_quadtree(lut, i0, i1, j0, j1, max_depth, mse_threshold, r_norm, c_norm):
    """Recursive quadtree partitioning with bilinear fits"""
    lut_region = lut[i0:i1, j0:j1]
    r_region = r_norm[i0:i1]
    c_region = c_norm[j0:j1]
    coeffs = fit_bilinear_plane(lut_region, r_region, c_region)
    a, b, c = coeffs

    # Compute mse for region
    r_pred, c_pred = np.meshgrid(r_region, c_region, indexing='ij')
    predicted = a + b * r_pred + c * c_pred
    mse_local = np.mean((lut_region - predicted)**2)

    # Subdivide if needed
    if (i1 - i0 > 4 and j1 - j0 > 4) and mse_local > mse_threshold and max_depth > 0:
        mid_i = (i0 + i1) // 2
        mid_j = (j0 + j1) // 2
        quadtree_regions = []
        for ii0, ii1 in [(i0, mid_i), (mid_i, i1)]:
            for jj0, jj1 in [(j0, mid_j), (mid_j, j1)]:
                quadtree_regions.extend(build_quadtree(lut, ii0, ii1, jj0, jj1, max_depth-1, mse_threshold, r_norm, c_norm))
        return quadtree_regions

    else:
        # Leaf, return region with coeffs
        return [(i0, i1, j0, j1, a, b, c)]

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

    # Chebyshev polynomial for analytical sheen modeling
    deg = 12
    terms = [(i, j) for i in range(deg + 1) for j in range(deg - i + 1)]

    n_terms = len(terms)
    print(f"Chebyshev, deg {deg}, total terms: {n_terms}")

    R_, C_ = np.meshgrid(r, c, indexing='ij')

    X = np.zeros((128 * 128, n_terms))
    k = 0
    for i, j in terms:
        X[:, k] = chebval(R_.ravel(), [0]*i + [1]) * chebval(C_.ravel(), [0]*j + [1])
        k += 1

    coeffs = np.linalg.lstsq(X, lut.ravel(), rcond=None)[0]
    coeffs = np.round(coeffs, 7)

    lut_approx = np.dot(X, coeffs).reshape(128, 128)
    mse = np.mean((lut - lut_approx)**2)
    ssim_val = ssim(lut, np.clip(lut_approx, 0, None), data_range=vmax)

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
