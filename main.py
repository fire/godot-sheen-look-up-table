import numpy as np
import sympy

def main():
    # Load data
    with open("thirdparty/sheen_lut_data.txt", "r") as f:
        content = f.read()

    # Parse the list
    data_str = content.strip()[1:-1]  # Remove [ and ]
    data = [float(val.strip()) for val in data_str.split(",")]

    # Reshape to 128x128
    lut = np.array(data).reshape(128, 128)

    # Normalized coordinates
    r = np.linspace(0, 1, 128)  # roughness perhaps
    c = np.linspace(0, 1, 128)  # cos_theta or something

    # Fit 2D polynomial of degree 8 for improved fit under 0.001 MSE
    deg = 8
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

    # Least squares fit
    coeffs, residuals, rank, s = np.linalg.lstsq(X, lut.ravel(), rcond=None)

    # Symbolic variables
    x, y = sympy.symbols('r cos_theta')  # assuming r is roughness, cos_theta is something

    # Build the expression
    expr = sum(coeffs[k] * x**i * y**j for k, (i, j) in enumerate(terms))

    print("Approximated analytical expression for the sheen LUT:")
    print(sympy.simplify(expr))

    # Optional: check the fit quality
    lut_approx = np.dot(X, coeffs).reshape(128, 128)
    mse = np.mean((lut - lut_approx)**2)
    print(f"Mean Squared Error: {mse}")

if __name__ == "__main__":
    main()
