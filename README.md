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
- **Fitting Method**: We fit a 2D polynomial of degree 4 using linear least squares optimization.
- **Output**: An analytical expression expressed symbolically using SymPy-derived equations.

## Current Approximation

Derivative of a degree 4 polynomial (15 terms) fitted using least squares, providing MSE ~0.056 for direct shader implementation without LUT lookup.

```
10.541436656511*cos_theta**4 - 15.8906606011087*cos_theta**3*r - 17.1488136680321*cos_theta**3 + 10.9806612332594*cos_theta**2*r**2 + 17.9790429861389*cos_theta**2*r + 8.53664845039853*cos_theta**2 + 0.134519806072525*cos_theta*r**3 - 12.0565363127585*cos_theta*r**2 - 3.74737790070742*cos_theta*r - 1.96569854338589*cos_theta - 0.0279244335528505*r**4 - 0.0341913862356636*r**3 + 2.29548431845381*r**2 - 0.149368395647259*r + 0.623898536703856
```

Where:
- `r`: Normalized roughness (0 to 1)
- `cos_theta`: NdotV (cosine of the viewing angle)

Mean Squared Error: 0.05590929665523771

## Usage

Run the approximation:

```bash
uv run python main.py
```

This will output the fitted expression and error metrics.

## Notes

This polynomial approximation provides a balance of accuracy (MSE 0.0559) and shader efficiency (15 terms, mostly multiplications and additions). Faster and simpler than higher-degree models but with reduced accuracy compared to LUT lookups.

## References

- Godot PR [#111568](https://github.com/godotengine/godot/pull/111568): Add sheen shading support
- Filament's sheen implementation
