// Mobile Sheen LUT analytical approximation.
// rank=4, degree=10, coefficients=88 + 5 scalar edge-correction parameters
// Expression: rank-4 separable Chebyshev over sqrt-warped inputs, plus a
// smooth branchless left-edge residual correction fitted against ground truth.

const int SHEEN_LUT_RANK = 4;
const int SHEEN_LUT_DEGREE = 10;
const float SHEEN_EDGE_AMPLITUDE = 0.75;
const float SHEEN_EDGE_ROUGH0 = 0.85;
const float SHEEN_EDGE_ROUGH1 = 1.0;
const float SHEEN_EDGE_COS0 = 0.02;
const float SHEEN_EDGE_COS1 = 0.0;
const float SHEEN_MOBILE_ROUGH_COEFFS[44] = float[44](2.79128730, 1.10049300, 0.352703300, -0.715766500, -0.737764100, -0.745066400, -0.737170400, -0.355291200, -0.535545300, -0.0911946000, -0.302863700, -3.38733960, -2.13088790, -0.256903000, -0.0953685000, 0.261723400, 0.108959100, -0.197772800, 0.0372126000, -0.449666300, 0.0176551000, -0.373570300, -0.379037600, 1.71684890, 0.417749800, 2.28086540, 1.62422130, 1.71906740, 2.04500910, 0.809181500, 1.86180280, 0.194989200, 1.20576770, -2.44473370, -2.31743920, -1.38393260, 0.00204440000, 0.246251700, 0.444987000, 0.582348700, 0.255443200, 0.524121600, 0.0668309000, 0.327211600);
const float SHEEN_MOBILE_COS_COEFFS[44] = float[44](1.65597770, -1.44873300, 0.832955200, -0.354371700, -0.307044600, 0.373793000, -0.142041800, -0.0307764000, 0.0784579000, -0.0550076000, 0.0230463000, 0.965232900, -0.958853900, 0.647891000, -0.272792300, -0.190386800, 0.245405400, -0.0985806000, -0.0137771000, 0.0465832000, -0.0334964000, 0.0141780000, 0.728540300, -0.829956400, 0.582212000, -0.316190100, -0.0221611000, 0.122872400, -0.0781959000, 0.0226701000, 0.00316890000, -0.00682420000, 0.00365600000, 0.169502600, 0.146289400, -0.183128100, 0.0727207000, -0.0822644000, 0.0518821000, -0.000424300000, -0.0266691000, 0.0275761000, -0.0163480000, 0.00633740000);

float sheen_mobile_cheb_eval(float x, const float coeffs[44], int offset) {
    float t0 = 1.0;
    float t1 = x;
    float acc = coeffs[offset] + coeffs[offset + 1] * t1;
    for (int n = 2; n <= SHEEN_LUT_DEGREE; n++) {
        float t2 = 2.0 * x * t1 - t0;
        acc += coeffs[offset + n] * t2;
        t0 = t1;
        t1 = t2;
    }
    return acc;
}

float sheen_lut_approx_mobile(float roughness, float cos_theta) {
    float roughness_clamped = clamp(roughness, 0.0, 1.0);
    float cos_theta_clamped = clamp(cos_theta, 0.0, 1.0);
    float r = 2.0 * sqrt(roughness_clamped) - 1.0;
    float v = 2.0 * sqrt(cos_theta_clamped) - 1.0;
    float y = 0.0;
    for (int k = 0; k < SHEEN_LUT_RANK; k++) {
        int offset = k * (SHEEN_LUT_DEGREE + 1);
        y += sheen_mobile_cheb_eval(r, SHEEN_MOBILE_ROUGH_COEFFS, offset) *
             sheen_mobile_cheb_eval(v, SHEEN_MOBILE_COS_COEFFS, offset);
    }
    float edge_weight = smoothstep(SHEEN_EDGE_ROUGH0, SHEEN_EDGE_ROUGH1, roughness_clamped) *
        smoothstep(SHEEN_EDGE_COS0, SHEEN_EDGE_COS1, cos_theta_clamped);
    y += SHEEN_EDGE_AMPLITUDE * edge_weight;
    return max(y, 0.0);
}
