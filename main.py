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
from scipy.optimize import least_squares

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

    # Degrees for numerator and denominator
    deg_num = 5  # degree for numerator polynomial
    deg_den = 3  # degree for denominator polynomial, keep lower to avoid overfitting

    # Generate terms for numerator
    terms_num = []
    for i in range(deg_num + 1):
        for j in range(deg_num + 1 - i):
            terms_num.append((i, j))
    n_terms_num = len(terms_num)

    # Generate terms for denominator (excluding (0,0) to avoid trivial solutions)
    terms_den = []
    for i in range(deg_den + 1):
        for j in range(deg_den + 1 - i):
            if i + j > 0:  # exclude constant term
                terms_den.append((i, j))
    n_terms_den = len(terms_den)

    total_terms = n_terms_num + n_terms_den

    R_, C_ = np.meshgrid(r, c, indexing='ij')
    R = R_.ravel()
    C = C_.ravel()
    lut_flat = lut.ravel()

    def residual(coeffs):
        numer_coeffs = coeffs[:n_terms_num]
        denom_coeffs = coeffs[n_terms_num:]

        p = np.zeros(len(lut_flat))
        for k, (i, j) in enumerate(terms_num):
            p += numer_coeffs[k] * R ** i * C ** j

        q = np.ones(len(lut_flat))
        for k, (i, j) in enumerate(terms_den):
            q += denom_coeffs[k] * R ** i * C ** j

        approx = p / q
        return lut_flat - approx

    # Initial guess
    initial_guess = np.zeros(total_terms)

    # Nonlinear least squares fit
    result = least_squares(residual, initial_guess, method='lm')  # Levenberg-Marquardt

    coeffs = result.x

    numer_coeffs = coeffs[:n_terms_num]
    denom_coeffs = coeffs[n_terms_num:]

    # Symbolic variables
    x, y_sym = sympy.symbols('r cos_theta')  # x is r, y_sym is cos_theta

    # Build numerator expression
    numer_expr = sum(numer_coeffs[k] * x**i * y_sym**j for k, (i, j) in enumerate(terms_num))

    # Denominator: 1 + sum denom terms (since we excluded (0,0))
    denom_expr = 1 + sum(denom_coeffs[k] * x**i * y_sym**j for k, (i, j) in enumerate(terms_den))

    # Rational function
    rational_expr = numer_expr / denom_expr
    rational_expr = sympy.simplify(rational_expr)

    print("Approximated analytical expression for the sheen LUT (Rational Function):")
    print(rational_expr)

    # Check fit quality
    p_values = np.zeros(len(lut_flat))
    for k, (i, j) in enumerate(terms_num):
        p_values += numer_coeffs[k] * R ** i * C ** j

    q_values = np.ones(len(lut_flat))
    for k, (i, j) in enumerate(terms_den):
        q_values += denom_coeffs[k] * R ** i * C ** j

    lut_approx = p_values / q_values
    lut_approx = lut_approx.reshape(128, 128)

    mse = np.mean((lut - lut_approx)**2)
    print(f"Mean Squared Error: {mse}")

if __name__ == "__main__":
    main()
