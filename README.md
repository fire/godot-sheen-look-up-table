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
- **Fitting Method**: The mobile expression is a rank-4 separable Chebyshev model, degree 10 per axis, over `sqrt`-warped shader inputs, with a smooth branchless residual correction for the bright left-edge region.
- **Output**: A generated analytical shader expression in `sheen_lut_approx.glsl`, plus a Lean4 proof artifact that checks the plausible witness DAG and evaluator.

## Current Approximation

The current approximation uses a rank-4, degree-10 mobile separable Chebyshev model plus a fitted left-edge residual correction. The final shader expression is:

```text
max(sum_k ChebR_k(2*sqrt(roughness)-1) * ChebC_k(2*sqrt(cos_theta)-1)
    + 0.75 * smoothstep(0.85, 1.0, roughness) * smoothstep(0.02, 0.0, cos_theta), 0.0)
```

The generated approximation metrics are:

- Chebyshev coefficients: 88
- Edge-correction scalars: 5
- Edge-corrected Mean Squared Error: 0.002349
- Edge-corrected Max Absolute Error: 1.953671

Where:
- `r`: Normalized roughness (0 to 1)
- `cos_theta`: NdotV (cosine of the viewing angle)

## Lean proof artifact

`SheenLutProof.lean` converts the rounded-coefficient model and edge correction into an exact rational Lean4 artifact. It uses a Flowref-style plausible witness DAG: every rank component is represented as a witness node, dependencies point only backward, Lean proves the graph is acyclic, and Lean proves that the witness DAG implements the generated separable Chebyshev expression plus edge correction after clamping negative values to zero.

Lake depends on `https://github.com/fire/plausible-witness-dag`; the proof artifact uses its trace/outcome vocabulary for the witness record.

The smoke executable reads `thirdparty/sheen_lut_data.txt` as ground truth and reports the expression's measured error against that file. The separable expression is the candidate renderer replacement; the LUT text file is the reference.

Verify the proof and run the smoke executable:

```bash
lake build sheen-lut-proof
.lake/build/bin/sheen-lut-proof
```

## Usage

Use the checked Lean4 artifact and generated GLSL approximation directly.

## Notes

This approximation is the mobile default: rank 4, degree 10, 88 Chebyshev coefficients, and 5 edge-correction scalars using sqrt-warped inputs and a low-rank separable Chebyshev model. The Lean4 proof checks the generated plausible witness DAG and evaluator structure.

## References

- Godot PR [#111568](https://github.com/godotengine/godot/pull/111568): Add sheen shading support
- Filament's sheen implementation
