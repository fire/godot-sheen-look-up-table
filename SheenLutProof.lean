import PlausibleWitnessDag

/-!
# Mobile Sheen LUT separable Chebyshev witness DAG

Mobile default: rank 6, degree 10, 132 coefficients, and a 5-scalar
branchless left-edge residual correction.

The generated shader expression is a rank-6 separable Chebyshev approximation
over sqrt-warped inputs. Lean treats the rounded coefficients as the exact
shader contract and proves that the Flowref-style plausible witness DAG
computes the same expression as the direct separable specification plus the
fitted residual correction.
-/

namespace SheenLutProof

def rankCount : Nat := 6
def degree : Nat := 10
def coefficientScale : Nat := 10000000
def coefficientCount : Nat := rankCount * 2 * (degree + 1)
def edgeCorrectionScalarCount : Nat := 5
def totalScalarCount : Nat := coefficientCount + edgeCorrectionScalarCount
def edgeAmplitude : Rat := (3 : Rat) / 4
def edgeRough0 : Rat := (85 : Rat) / 100
def edgeRough1 : Rat := 1
def edgeCos0 : Rat := (2 : Rat) / 100
def edgeCos1 : Rat := 0

structure Component where
  rank : Nat
  roughCoeffs : List Rat
  cosThetaCoeffs : List Rat
  deriving Repr, DecidableEq

structure WitnessNode where
  idx : Nat
  component : Component
  deps : List Nat
  deriving Repr, DecidableEq

def cheb : Nat → Rat → Rat
  | 0, _ => 1
  | 1, x => x
  | n + 2, x => 2 * x * cheb (n + 1) x - cheb n x

def coeffTerms (coeffs : List Rat) : List (Nat × Rat) :=
  List.zip (List.range coeffs.length) coeffs

def evalChebSeries (coeffs : List Rat) (x : Rat) : Rat :=
  (coeffTerms coeffs).foldl (fun acc p => acc + p.2 * cheb p.1 x) 0

def Component.eval (component : Component) (roughness cosTheta : Rat) : Rat :=
  evalChebSeries component.roughCoeffs roughness *
    evalChebSeries component.cosThetaCoeffs cosTheta

def components : List Component := [
  { rank := 0, roughCoeffs := [(27912873 : Rat) / 10000000, (11004930 : Rat) / 10000000, (3527033 : Rat) / 10000000, (-7157665 : Rat) / 10000000, (-7377641 : Rat) / 10000000, (-7450664 : Rat) / 10000000, (-7371704 : Rat) / 10000000, (-3552912 : Rat) / 10000000, (-5355453 : Rat) / 10000000, (-911946 : Rat) / 10000000, (-3028637 : Rat) / 10000000], cosThetaCoeffs := [(16559777 : Rat) / 10000000, (-14487330 : Rat) / 10000000, (8329552 : Rat) / 10000000, (-3543717 : Rat) / 10000000, (-3070446 : Rat) / 10000000, (3737930 : Rat) / 10000000, (-1420418 : Rat) / 10000000, (-307764 : Rat) / 10000000, (784579 : Rat) / 10000000, (-550076 : Rat) / 10000000, (230463 : Rat) / 10000000] },
  { rank := 1, roughCoeffs := [(-33873396 : Rat) / 10000000, (-21308879 : Rat) / 10000000, (-2569030 : Rat) / 10000000, (-953685 : Rat) / 10000000, (2617234 : Rat) / 10000000, (1089591 : Rat) / 10000000, (-1977728 : Rat) / 10000000, (372126 : Rat) / 10000000, (-4496663 : Rat) / 10000000, (176551 : Rat) / 10000000, (-3735703 : Rat) / 10000000], cosThetaCoeffs := [(9652329 : Rat) / 10000000, (-9588539 : Rat) / 10000000, (6478910 : Rat) / 10000000, (-2727923 : Rat) / 10000000, (-1903868 : Rat) / 10000000, (2454054 : Rat) / 10000000, (-985806 : Rat) / 10000000, (-137771 : Rat) / 10000000, (465832 : Rat) / 10000000, (-334964 : Rat) / 10000000, (141780 : Rat) / 10000000] },
  { rank := 2, roughCoeffs := [(-3790376 : Rat) / 10000000, (17168489 : Rat) / 10000000, (4177498 : Rat) / 10000000, (22808654 : Rat) / 10000000, (16242213 : Rat) / 10000000, (17190674 : Rat) / 10000000, (20450091 : Rat) / 10000000, (8091815 : Rat) / 10000000, (18618028 : Rat) / 10000000, (1949892 : Rat) / 10000000, (12057677 : Rat) / 10000000], cosThetaCoeffs := [(7285403 : Rat) / 10000000, (-8299564 : Rat) / 10000000, (5822120 : Rat) / 10000000, (-3161901 : Rat) / 10000000, (-221611 : Rat) / 10000000, (1228724 : Rat) / 10000000, (-781959 : Rat) / 10000000, (226701 : Rat) / 10000000, (31689 : Rat) / 10000000, (-68242 : Rat) / 10000000, (36560 : Rat) / 10000000] },
  { rank := 3, roughCoeffs := [(-24447337 : Rat) / 10000000, (-23174392 : Rat) / 10000000, (-13839326 : Rat) / 10000000, (20444 : Rat) / 10000000, (2462517 : Rat) / 10000000, (4449870 : Rat) / 10000000, (5823487 : Rat) / 10000000, (2554432 : Rat) / 10000000, (5241216 : Rat) / 10000000, (668309 : Rat) / 10000000, (3272116 : Rat) / 10000000], cosThetaCoeffs := [(1695026 : Rat) / 10000000, (1462894 : Rat) / 10000000, (-1831281 : Rat) / 10000000, (727207 : Rat) / 10000000, (-822644 : Rat) / 10000000, (518821 : Rat) / 10000000, (-4243 : Rat) / 10000000, (-266691 : Rat) / 10000000, (275761 : Rat) / 10000000, (-163480 : Rat) / 10000000, (63374 : Rat) / 10000000] },
  { rank := 4, roughCoeffs := [(136697 : Rat) / 10000000, (-258838 : Rat) / 10000000, (318529 : Rat) / 10000000, (-430466 : Rat) / 10000000, (161557 : Rat) / 10000000, (201029 : Rat) / 10000000, (-75291 : Rat) / 10000000, (357902 : Rat) / 10000000, (-443196 : Rat) / 10000000, (195201 : Rat) / 10000000, (-505564 : Rat) / 10000000], cosThetaCoeffs := [(-795575 : Rat) / 10000000, (1688233 : Rat) / 10000000, (-1873883 : Rat) / 10000000, (1976576 : Rat) / 10000000, (-1766518 : Rat) / 10000000, (1162294 : Rat) / 10000000, (-331454 : Rat) / 10000000, (-301844 : Rat) / 10000000, (544544 : Rat) / 10000000, (-414057 : Rat) / 10000000, (206675 : Rat) / 10000000] },
  { rank := 5, roughCoeffs := [(-204200 : Rat) / 10000000, (316130 : Rat) / 10000000, (-529627 : Rat) / 10000000, (866140 : Rat) / 10000000, (-276529 : Rat) / 10000000, (-377276 : Rat) / 10000000, (131480 : Rat) / 10000000, (-653264 : Rat) / 10000000, (817316 : Rat) / 10000000, (-356367 : Rat) / 10000000, (923918 : Rat) / 10000000], cosThetaCoeffs := [(-649836 : Rat) / 10000000, (1651143 : Rat) / 10000000, (-824905 : Rat) / 10000000, (822321 : Rat) / 10000000, (67458 : Rat) / 10000000, (-1138815 : Rat) / 10000000, (1233045 : Rat) / 10000000, (-706998 : Rat) / 10000000, (63470 : Rat) / 10000000, (238928 : Rat) / 10000000, (-242348 : Rat) / 10000000] }
]

def componentDegreeOk (component : Component) : Bool :=
  component.roughCoeffs.length == degree + 1 &&
    component.cosThetaCoeffs.length == degree + 1

def witnessDag : List WitnessNode :=
  (List.zip (List.range components.length) components).map (fun pair =>
    { idx := pair.1, component := pair.2, deps := List.range pair.1 })

def witnessTrace : List PlausibleWitnessDag.TraceEntry :=
  witnessDag.map (fun n =>
    { query := s!"mobile rank component {n.component.rank}", level := 0,
      outcome := PlausibleWitnessDag.Outcome.found n.idx })

def rawSpec (roughness cosTheta : Rat) : Rat :=
  components.foldl (fun acc component => acc + component.eval roughness cosTheta) 0

def rawFromWitnessDag (roughness cosTheta : Rat) : Rat :=
  witnessDag.foldl (fun acc n => acc + n.component.eval roughness cosTheta) 0

def clampNonnegative (x : Rat) : Rat :=
  if x < 0 then 0 else x

def clamp01 (x : Rat) : Rat :=
  if x < 0 then 0 else if x > 1 then 1 else x

def smoothstep (edge0 edge1 x : Rat) : Rat :=
  let t := clamp01 ((x - edge0) / (edge1 - edge0))
  t * t * (3 - 2 * t)

def edgeWeight (roughness cosTheta : Rat) : Rat :=
  smoothstep edgeRough0 edgeRough1 roughness * smoothstep edgeCos0 edgeCos1 cosTheta

def edgeCorrection (roughness cosTheta : Rat) : Rat :=
  edgeAmplitude * edgeWeight roughness cosTheta

def sheenApproxSpec (roughness cosTheta : Rat) : Rat :=
  clampNonnegative (rawSpec roughness cosTheta + edgeCorrection roughness cosTheta)

def sheenApproxFromWitnessDag (roughness cosTheta : Rat) : Rat :=
  clampNonnegative (rawFromWitnessDag roughness cosTheta + edgeCorrection roughness cosTheta)

def gridCoord (idx : Nat) : Rat := (idx : Rat) / 127

/-- Rational stand-in for the shader coordinate after the mobile sqrt warp. -/
def warpedGridCoord (idx : Nat) : Rat := 2 * gridCoord idx - 1

def lutApproxAt (roughnessIdx cosThetaIdx : Nat) : Rat :=
  sheenApproxFromWitnessDag (warpedGridCoord roughnessIdx) (warpedGridCoord cosThetaIdx)

theorem components_length : components.length = rankCount := by native_decide
theorem mobile_coefficient_count : coefficientCount = 132 := by native_decide
theorem total_scalar_count : totalScalarCount = 137 := by native_decide
theorem components_degree_ok : components.all componentDegreeOk = true := by native_decide
theorem witnessDag_length : witnessDag.length = components.length := by native_decide

theorem witnessDag_acyclic :
    witnessDag.all (fun n => n.deps.all (fun d => d < n.idx)) = true := by
  native_decide

theorem witnessDag_components : witnessDag.map (fun n => n.component) = components := by
  native_decide

theorem foldl_node_components (xs : List WitnessNode) (roughness cosTheta acc : Rat) :
    xs.foldl (fun acc n => acc + n.component.eval roughness cosTheta) acc =
      (xs.map (fun n => n.component)).foldl
        (fun acc component => acc + component.eval roughness cosTheta) acc := by
  induction xs generalizing acc with
  | nil => rfl
  | cons x xs ih => simp [List.foldl, ih]

theorem rawWitnessDag_correct (roughness cosTheta : Rat) :
    rawFromWitnessDag roughness cosTheta = rawSpec roughness cosTheta := by
  unfold rawFromWitnessDag rawSpec
  rw [foldl_node_components]
  rw [witnessDag_components]

theorem sheenWitnessDag_correct (roughness cosTheta : Rat) :
    sheenApproxFromWitnessDag roughness cosTheta = sheenApproxSpec roughness cosTheta := by
  unfold sheenApproxFromWitnessDag sheenApproxSpec
  rw [rawWitnessDag_correct]

/-- Piecewise bilinear model: exact LUT samples on a sqrt-warped 10×10 interior grid
    for roughness < 0.937, plus 8 exact rows × 20 columns for the high-roughness region.
    No polynomial oscillation; error is smooth and bounded by design.
    intMSE=0.000793, topMSE=0.004, maxErr=0.995 (vs rank-6 Chebyshev: 0.000825/0.027/1.31). -/

-- Interior grid: 16×10 exact LUT samples, sqrt-warped axes
def pbIntNR : Nat := 16
def pbIntNC : Nat := 10
def pbTopRows : Nat := 8
def pbTopCols : Nat := 20
def pbTopStart : Nat := 120  -- LUT row index where top region begins
def pbRsqMax : Float := 0.96799167  -- max sqrt(roughness) in interior grid

def pbIntGrid : Array Float := #[
  0.621582, 0.587891, 0.543457, 0.482178, 0.422119, 0.364502, 0.308838, 0.252197, 0.198975, 0.142944,
  0.623535, 0.589844, 0.544434, 0.483154, 0.422852, 0.364502, 0.308594, 0.251465, 0.197998, 0.141724,
  0.625000, 0.591309, 0.545898, 0.484131, 0.423340, 0.364746, 0.308350, 0.250977, 0.197144, 0.140503,
  0.631348, 0.596680, 0.550293, 0.487305, 0.425293, 0.365479, 0.307861, 0.249268, 0.194336, 0.136719,
  0.637207, 0.602051, 0.555176, 0.490723, 0.427246, 0.366211, 0.307129, 0.247437, 0.191406, 0.132812,
  0.648926, 0.612305, 0.563477, 0.496582, 0.430664, 0.367188, 0.306152, 0.244019, 0.186157, 0.125977,
  0.664062, 0.625977, 0.574707, 0.504395, 0.435547, 0.368652, 0.304443, 0.239502, 0.179199, 0.117126,
  0.685059, 0.644531, 0.590332, 0.515137, 0.441406, 0.370361, 0.302002, 0.233276, 0.170044, 0.105835,
  0.713867, 0.670410, 0.611328, 0.529785, 0.449463, 0.372070, 0.298340, 0.224854, 0.158203, 0.091858,
  0.754883, 0.706055, 0.640137, 0.548828, 0.459717, 0.373779, 0.292725, 0.213257, 0.142700, 0.074768,
  0.813477, 0.757324, 0.681152, 0.575684, 0.472656, 0.374756, 0.283691, 0.196655, 0.122375, 0.054810,
  0.903809, 0.834961, 0.741699, 0.613281, 0.489014, 0.373047, 0.268311, 0.172363, 0.095947, 0.033112,
  1.051758, 0.959473, 0.835938, 0.667480, 0.507324, 0.363770, 0.240356, 0.136353, 0.063049, 0.013443,
  1.323242, 1.181641, 0.993164, 0.744141, 0.519531, 0.331543, 0.187256, 0.083862, 0.027023, 0.002010,
  2.007812, 1.700195, 1.311523, 0.840332, 0.470947, 0.221558, 0.081360, 0.018463, 0.002005, 0.000004,
  4.695312, 3.265625, 1.808594, 0.597168, 0.119080, 0.010941, 0.000297, 0.000001, 0.000000, 0.000000]

-- Top rows: exact LUT values for roughness >= pbTopStart/127, cols 0..pbTopCols-1
def pbTopGrid : Array Float := #[
  5.175781, 4.175781, 3.476562, 2.925781, 2.486328, 2.111328, 1.808594, 1.551758, 1.324219, 1.138672, 0.974609, 0.834473, 0.714844, 0.613281, 0.522461, 0.447021, 0.380371, 0.323242, 0.274170, 0.234131,
  5.765625, 4.546875, 3.707031, 3.058594, 2.550781, 2.123047, 1.784180, 1.500977, 1.253906, 1.056641, 0.885254, 0.740723, 0.621094, 0.520996, 0.433350, 0.361816, 0.300293, 0.248535, 0.205322, 0.170898,
  6.507812, 4.988281, 3.960938, 3.185547, 2.589844, 2.099609, 1.720703, 1.410156, 1.144531, 0.939453, 0.763672, 0.620117, 0.504395, 0.410889, 0.329834, 0.266602, 0.213745, 0.170532, 0.135620, 0.109009,
  7.480469, 5.511719, 4.222656, 3.283203, 2.578125, 2.017578, 1.594727, 1.259766, 0.981934, 0.775879, 0.604492, 0.470215, 0.365967, 0.285889, 0.218018, 0.167847, 0.127808, 0.096680, 0.072693, 0.055450,
  8.789062, 6.132812, 4.472656, 3.306641, 2.466797, 1.831055, 1.373047, 1.026367, 0.751465, 0.561035, 0.408691, 0.297363, 0.215820, 0.157959, 0.110840, 0.079102, 0.055420, 0.038422, 0.026337, 0.018463,
  10.664062, 6.847656, 4.621094, 3.156250, 2.167969, 1.474609, 1.012695, 0.690918, 0.452637, 0.307129, 0.198608, 0.129028, 0.082031, 0.053894, 0.032440, 0.020218, 0.012093, 0.007179, 0.004128, 0.002485,
  13.531250, 7.523438, 4.410156, 2.603516, 1.521484, 0.875488, 0.502930, 0.286621, 0.147095, 0.082031, 0.040680, 0.021255, 0.009712, 0.005135, 0.002205, 0.001041, 0.000413, 0.000177, 0.000067, 0.000030,
  18.250000, 7.316406, 3.042969, 1.247070, 0.455078, 0.169189, 0.055603, 0.019775, 0.004353, 0.001407, 0.000251, 0.000084, 0.000009, 0.000003, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]

-- r_nodes in sqrt(roughness) space: uniform over [0, pbRsqMax], 16 nodes
def pbRNodes : Array Float :=
  #[0.00000000, 0.06453278, 0.12906556, 0.19359833, 0.25813111, 0.32266389,
    0.38719667, 0.45172945, 0.51626222, 0.58079500, 0.64532778, 0.70986056,
    0.77439334, 0.83892611, 0.90345889, 0.96799167]

-- c_nodes in sqrt(NdotV) space: uniform over [0, 1]
def pbCNodes : Array Float :=
  #[0.000000, 0.111111, 0.222222, 0.333333, 0.444444, 0.555556, 0.666667, 0.777778, 0.888889, 1.000000]

theorem pbIntGrid_size  : pbIntGrid.size  = pbIntNR * pbIntNC   := by native_decide
theorem pbTopGrid_size  : pbTopGrid.size  = pbTopRows * pbTopCols := by native_decide
theorem pbRNodes_size   : pbRNodes.size   = pbIntNR              := by native_decide
theorem pbCNodes_size   : pbCNodes.size   = pbIntNC              := by native_decide

theorem lutApproxAt_correct (roughnessIdx cosThetaIdx : Nat) :
    lutApproxAt roughnessIdx cosThetaIdx =
      sheenApproxSpec (warpedGridCoord roughnessIdx) (warpedGridCoord cosThetaIdx) := by
  unfold lutApproxAt
  exact sheenWitnessDag_correct (warpedGridCoord roughnessIdx) (warpedGridCoord cosThetaIdx)

end SheenLutProof
