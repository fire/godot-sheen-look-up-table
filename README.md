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
- **Fitting Method**: We fit a 2D polynomial of degree 8 using linear least squares optimization.
- **Output**: An analytical expression expressed symbolically using SymPy-derived equations.

## Current Approximation

Derivative of a degree 8 polynomial (45 terms) fitted using least squares, providing MSE ~0.001 for direct shader implementation without LUT lookup.

```
-557.091779271558*cos_theta**8 + 464.017389846217*cos_theta**7*r + 2152.1266164332*cos_theta**7 - 230.321684192249*cos_theta**6*r**2 - 1525.96376250796*cos_theta**6*r - 3384.23333420742*cos_theta**6 + 2.86174085877387*cos_theta**5*r**3 + 744.890840615195*cos_theta**5*r**2 + 1932.42802940127*cos_theta**5*r + 2787.74835270357*cos_theta**5 + 104.289164776309*cos_theta**4*r**4 - 194.098783820854*cos_theta**4*r**3 - 825.798219330147*cos_theta**4*r**2 - 1189.47390170691*cos_theta**4*r - 1289.70446461101*cos_theta**4 + 16.4307961138826*cos_theta**3*r**5 - 264.236227330607*cos_theta**3*r**4 + 435.698733067964*cos_theta**3*r**3 + 330.115180358558*cos_theta**3*r**2 + 377.760815347863*cos_theta**3*r + 333.246911023428*cos_theta**3 - 33.7673457364751*cos_theta**2*r**6 + 56.2723698949418*cos_theta**2*r**5 + 148.016374822698*cos_theta**2*r**4 - 297.359389044023*cos_theta**2*r**3 - 3.59259176216401*cos_theta**2*r**2 - 64.9201817967158*cos_theta**2*r - 44.5916577589042*cos_theta**2 - 12.0589143469186*cos_theta*r**7 + 79.3217215327783*cos_theta*r**6 - 133.214050066957*cos_theta*r**5 + 45.9217563239256*cos_theta*r**4 + 45.1941989068905*cos_theta*r**3 - 14.9524287804133*cos_theta*r**2 + 6.30200063105549*cos_theta*r + 2.44540720268555*cos_theta + 8.53860911397782*r**8 - 24.68454951197*r**7 + 15.367948662591*r**6 + 11.3701055151527*r**5 - 13.5672939003094*r**4 + 2.94229103229011*r**3 + 0.453115385148636*r**2 - 0.128715903958555*r - 0.00103032751274113
```

Where:
- `r`: Normalized roughness (0 to 1)
- `cos_theta`: NdotV (cosine of the viewing angle)

Mean Squared Error: 0.000967 (under 0.001 threshold for shader performance)

## Usage

Run the approximation:

```bash
uv run python main.py
```

This will output the fitted expression and error metrics.

## Notes

This polynomial approximation provides a balance of accuracy (MSE 0.000967) and shader efficiency (45 terms, mostly multiplications and additions). Slower than low-degree models but faster than LUT lookups for mobile/compatibility renderers.

## References

- Godot PR #111568: Add sheen shading support
- Filament's sheen implementation
