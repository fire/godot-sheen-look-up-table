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

theorem lutApproxAt_correct (roughnessIdx cosThetaIdx : Nat) :
    lutApproxAt roughnessIdx cosThetaIdx =
      sheenApproxSpec (warpedGridCoord roughnessIdx) (warpedGridCoord cosThetaIdx) := by
  unfold lutApproxAt
  exact sheenWitnessDag_correct (warpedGridCoord roughnessIdx) (warpedGridCoord cosThetaIdx)

end SheenLutProof
