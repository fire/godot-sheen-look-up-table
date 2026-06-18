# Sheen LUT Analytical Approximation

## User Scenario

This project addresses a need in Godot Engine's rendering system, specifically for mobile and compatibility renderers that lack support for DFG (Distribution-Function Geometry) Lookup Tables. In the context of Godot Pull Request #111568 ("Add sheen shading support"), sheen shading was introduced to render cloth materials like cotton, velvet, and silk more realistically.

Sheen shading uses a DFG LUT for pre-filtered environment lighting, approximating the Distribution-Function Geometry integral (brdf.z/cloth_brdf). However, mobile and compatibility renderers cannot use this LUT due to performance constraints, leading to artifacts, especially at low roughness values (<= 0.3).

This project provides an analytical approximation of the Sheen LUT (blue channel of dfg_lut.dds) using numerical fitting and symbolic expression generation. The goal is to replace the texture lookup with a fast, compute-efficient analytical function that can run on all renderers.

Variables (from integrate_dfg.glsl in the PR):
- `r`: Sheen roughness (0 to 1)
- `cos_theta`: NdotV (cosine of viewing angle, 0 to 1)

## Implementation

- **Data Source**: The `thirdparty/sheen_lut_data.txt` contains the blue channel values extracted from the DFG DDS file, forming a 128x128 lookup table.
- **Fitting Method**: We fit `sqrt(LUT)` with a rank-8 separable Chebyshev model, degree 20 per axis, using SVD plus univariate Chebyshev fits.
- **Output**: A generated analytical shader expression in `sheen_lut_approx.glsl`, plus a Lean4 proof artifact that checks the plausible witness DAG and evaluator.

## Current Approximation

The current approximation uses a rank-8, degree-20 separable Chebyshev model in square-root space. The final shader expression is:

```text
square(max(sum_k ChebR_k(roughness) * ChebC_k(cos_theta), 0.0))
```

The generated approximation metrics are:

- Coefficients: 336
- Mean Squared Error: 0.0000054600
- Max Absolute Error: 0.1602879545
- Structural Similarity Index (SSIM): 0.9999983

Where:
- `r`: Normalized roughness (0 to 1)
- `cos_theta`: NdotV (cosine of the viewing angle)

## Lean proof artifact

`SheenLutProof.lean` converts the fitted rounded-coefficient model into an exact rational Lean4 artifact. It uses a Flowref-style plausible witness DAG: every rank component is represented as a witness node, dependencies point only backward, Lean proves the graph is acyclic, and Lean proves that the witness DAG implements the generated separable Chebyshev expression after clamp-and-square.

Verify the proof and run the smoke executable:

```bash
lake build sheen-lut-proof
.lake/build/bin/sheen-lut-proof
```

## Usage

Use the checked Lean4 artifact and generated GLSL approximation directly.

## Notes

This approximation replaces the previous direct 2D polynomial fit with a low-rank square-root-space model. The Lean4 proof checks the generated plausible witness DAG and evaluator structure.

## References

- Godot PR [#111568](https://github.com/godotengine/godot/pull/111568): Add sheen shading support
- Filament's sheen implementation
