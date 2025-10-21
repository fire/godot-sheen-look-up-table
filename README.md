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
- **Fitting Method**: We fit a rational function (ratio of 2D polynomials, numerator degree 5, denominator degree 3) using nonlinear least squares optimization.
- **Output**: An analytical expression expressed symbolically using SymPy-derived equations.

## Current Approximation

Rational function fitted using nonlinear least squares, providing superior accuracy for shader implementation without LUT lookup.

Numerator (degree 5 polynomial, 21 terms):

```
1.10125048040805*cos_theta**5 + 6.74489226245854*cos_theta**4*r - 8.39998369444234*cos_theta**4 + 2.39569908771151*cos_theta**3*r**2 - 16.854682875742*cos_theta**3*r + 15.0376015207074*cos_theta**3 - 1.25628927025611*cos_theta**2*r**3 - 2.46572927745832*cos_theta**2*r**2 + 16.012008976718*cos_theta**2*r - 12.1524412563849*cos_theta**2 + 3.07814209710205*cos_theta*r**4 - 5.79768979753759*cos_theta*r**3 + 7.71629423145735*cos_theta*r**2 - 10.6212176237166*cos_theta*r + 5.55862212332412*cos_theta + 1.22191813729662*r**5 - 3.57850307085723*r**4 + 3.67267227650517*r**3 - 1.14670718891106*r**2 - 0.759436562230116*r + 0.592867865882987
```

Denominator (degree 3 polynomial + 1, 10 terms):

```
5.71239688081368*cos_theta**3 + 7.33391521962325*cos_theta**2*r - 6.60743404579406*cos_theta**2 + 11.2231070777972*cos_theta*r**2 - 23.1288976628828*cos_theta*r + 11.9206921626418*cos_theta - 0.250515373814049*r**3 + 1.48203243327513*r**2 - 2.2313629253502*r + 1
```

Where:

- `r`: Normalized roughness (0 to 1)
- `cos_theta`: NdotV (cosine of the viewing angle)

Mean Squared Error: 5.337380077659139e-05

## Usage

Run the approximation:

```bash
uv run python main.py
```

This will output the fitted expression and error metrics.

## Notes

This rational function approximation provides higher accuracy (MSE ~0.00005) than polynomial approximations, achieving ~20x better fidelity while maintaining reasonable shader complexity. The expression involves a division operation but is efficient for real-time rendering and superior to very high-degree polynomial fits when a managed analytical expression is needed.

## References

- Godot PR [#111568](https://github.com/godotengine/godot/pull/111568): Add sheen shading support
- Filament's sheen implementation
