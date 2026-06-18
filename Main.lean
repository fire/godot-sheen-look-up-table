import SheenLutProof
import Lean

open SheenLutProof

namespace SheenLutMobileCheck

def lutPath : System.FilePath := "thirdparty/sheen_lut_data.txt"

def roughCoeffs : Array Float := #[
  2.79128730, 1.10049300, 0.352703300, -0.715766500, -0.737764100,
  -0.745066400, -0.737170400, -0.355291200, -0.535545300, -0.0911946000,
  -0.302863700, -3.38733960, -2.13088790, -0.256903000, -0.0953685000,
  0.261723400, 0.108959100, -0.197772800, 0.0372126000, -0.449666300,
  0.0176551000, -0.373570300, -0.379037600, 1.71684890, 0.417749800,
  2.28086540, 1.62422130, 1.71906740, 2.04500910, 0.809181500,
  1.86180280, 0.194989200, 1.20576770, -2.44473370, -2.31743920,
  -1.38393260, 0.00204440000, 0.246251700, 0.444987000, 0.582348700,
  0.255443200, 0.524121600, 0.0668309000, 0.327211600]

def cosCoeffs : Array Float := #[
  1.65597770, -1.44873300, 0.832955200, -0.354371700, -0.307044600,
  0.373793000, -0.142041800, -0.0307764000, 0.0784579000, -0.0550076000,
  0.0230463000, 0.965232900, -0.958853900, 0.647891000, -0.272792300,
  -0.190386800, 0.245405400, -0.0985806000, -0.0137771000, 0.0465832000,
  -0.0334964000, 0.0141780000, 0.728540300, -0.829956400, 0.582212000,
  -0.316190100, -0.0221611000, 0.122872400, -0.0781959000, 0.0226701000,
  0.00316890000, -0.00682420000, 0.00365600000, 0.169502600, 0.146289400,
  -0.183128100, 0.0727207000, -0.0822644000, 0.0518821000, -0.000424300000,
  -0.0266691000, 0.0275761000, -0.0163480000, 0.00633740000]

def clamp01 (x : Float) : Float := max 0.0 (min 1.0 x)

def warpedCoord (idx : Nat) : Float :=
  2.0 * Float.sqrt (clamp01 ((Float.ofNat idx) / 127.0)) - 1.0

/-- Precomputed float [0,1] coordinates for grid indices 0..127. -/
def gridFloats : Array Float := Id.run do
  let mut arr := Array.mkEmpty 128
  for i in [0:128] do
    arr := arr.push ((Float.ofNat i) / 127.0)
  pure arr

/-- Precomputed warped (sqrt-mapped, [-1,1]) coordinates for grid indices 0..127. -/
def warpedFloats : Array Float := Id.run do
  let mut arr := Array.mkEmpty 128
  for i in [0:128] do
    arr := arr.push (warpedCoord i)
  pure arr

def chebEval (coeffs : Array Float) (offset : Nat) (x : Float) : Float :=
  let t0 := 1.0
  let t1 := x
  let acc := coeffs[offset]! + coeffs[offset + 1]! * t1
  Id.run do
    let mut t0 := t0
    let mut t1 := t1
    let mut acc := acc
    for n in [2:11] do
      let t2 := 2.0 * x * t1 - t0
      acc := acc + coeffs[offset + n]! * t2
      t0 := t1
      t1 := t2
    pure acc

def approxAt (roughnessIdx cosThetaIdx : Nat) : Float :=
  let r := warpedCoord roughnessIdx
  let v := warpedCoord cosThetaIdx
  let y :=
    chebEval roughCoeffs 0 r * chebEval cosCoeffs 0 v +
    chebEval roughCoeffs 11 r * chebEval cosCoeffs 11 v +
    chebEval roughCoeffs 22 r * chebEval cosCoeffs 22 v +
    chebEval roughCoeffs 33 r * chebEval cosCoeffs 33 v
  max y 0.0

/-- Precompute the 128×128 grid of approxAt values once, reused across all edge candidates. -/
def buildApproxGrid : Array Float :=
  Id.run do
    let mut arr := Array.mkEmpty (128 * 128)
    for i in [0:128] do
      let r := warpedFloats[i]!
      for j in [0:128] do
        let v := warpedFloats[j]!
        let y :=
          chebEval roughCoeffs 0 r * chebEval cosCoeffs 0 v +
          chebEval roughCoeffs 11 r * chebEval cosCoeffs 11 v +
          chebEval roughCoeffs 22 r * chebEval cosCoeffs 22 v +
          chebEval roughCoeffs 33 r * chebEval cosCoeffs 33 v
        arr := arr.push (max y 0.0)
    pure arr

structure EdgeParams where
  rough0 : Float
  rough1 : Float
  cos0 : Float
  cos1 : Float
  deriving Repr

structure EdgeFit where
  params : EdgeParams
  amplitude : Float
  mse : Float
  maxErr : Float
  maxIdx : Nat
  deriving Repr

instance : Inhabited EdgeParams where
  default := { rough0 := 0.75, rough1 := 1.0, cos0 := 0.08, cos1 := 0.0 }

instance : Inhabited EdgeFit where
  default := { params := default, amplitude := 0.0, mse := 0.0, maxErr := 0.0, maxIdx := 0 }

def smoothstep (edge0 edge1 x : Float) : Float :=
  let t := clamp01 ((x - edge0) / (edge1 - edge0))
  t * t * (3.0 - 2.0 * t)

def edgeWeight (p : EdgeParams) (roughness cosTheta : Float) : Float :=
  smoothstep p.rough0 p.rough1 roughness * smoothstep p.cos0 p.cos1 cosTheta

def edgeParamCandidates : Array EdgeParams := #[
  { rough0 := 0.55, rough1 := 0.95, cos0 := 0.02, cos1 := 0.0 },
  { rough0 := 0.55, rough1 := 0.95, cos0 := 0.04, cos1 := 0.0 },
  { rough0 := 0.55, rough1 := 0.95, cos0 := 0.08, cos1 := 0.0 },
  { rough0 := 0.55, rough1 := 0.95, cos0 := 0.12, cos1 := 0.0 },
  { rough0 := 0.65, rough1 := 0.95, cos0 := 0.02, cos1 := 0.0 },
  { rough0 := 0.65, rough1 := 0.95, cos0 := 0.04, cos1 := 0.0 },
  { rough0 := 0.65, rough1 := 0.95, cos0 := 0.08, cos1 := 0.0 },
  { rough0 := 0.65, rough1 := 0.95, cos0 := 0.12, cos1 := 0.0 },
  { rough0 := 0.75, rough1 := 1.0, cos0 := 0.02, cos1 := 0.0 },
  { rough0 := 0.75, rough1 := 1.0, cos0 := 0.04, cos1 := 0.0 },
  { rough0 := 0.75, rough1 := 1.0, cos0 := 0.08, cos1 := 0.0 },
  { rough0 := 0.75, rough1 := 1.0, cos0 := 0.12, cos1 := 0.0 },
  { rough0 := 0.85, rough1 := 1.0, cos0 := 0.02, cos1 := 0.0 },
  { rough0 := 0.85, rough1 := 1.0, cos0 := 0.04, cos1 := 0.0 },
  { rough0 := 0.85, rough1 := 1.0, cos0 := 0.08, cos1 := 0.0 },
  { rough0 := 0.85, rough1 := 1.0, cos0 := 0.12, cos1 := 0.0 }]

def edgeAmplitudeCandidates : Array Float := #[
  -1.00, -0.75, -0.50, -0.25, 0.25, 0.50, 0.75, 1.00,
  1.25, 1.50, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50]

def edgeCandidateCount : Nat := edgeParamCandidates.size * edgeAmplitudeCandidates.size

def edgeCandidateAt (idx : Nat) : EdgeParams × Float :=
  let p := edgeParamCandidates[idx / edgeAmplitudeCandidates.size]!
  let amp := edgeAmplitudeCandidates[idx % edgeAmplitudeCandidates.size]!
  (p, amp)

def evaluateEdgeFit (lut approxGrid : Array Float) (p : EdgeParams) (amp : Float) : EdgeFit :=
  Id.run do
    let mut sumSq := 0.0
    let mut maxErr := 0.0
    let mut maxIdx := 0
    for i in [0:128] do
      let roughness := gridFloats[i]!
      for j in [0:128] do
        let idx := i * 128 + j
        let cosTheta := gridFloats[j]!
        let y := max (approxGrid[idx]! + amp * edgeWeight p roughness cosTheta) 0.0
        let err := Float.abs (lut[idx]! - y)
        sumSq := sumSq + err * err
        if err > maxErr then
          maxErr := err
          maxIdx := idx
    pure (EdgeFit.mk p amp (sumSq / 16384.0) maxErr maxIdx)

def chooseBetter (a b : EdgeFit) : EdgeFit :=
  if a.maxErr < b.maxErr then a
  else if b.maxErr < a.maxErr then b
  else if a.mse <= b.mse then a
  else b

def fitEdgeCorrection (lut approxGrid : Array Float) : IO (EdgeFit × PlausibleWitnessDag.TraceEntry) := do
  let levels : Array PlausibleWitnessDag.Level := #[
    { idx := 0, walkSteps := edgeCandidateCount, finBound := 256, numInst := 64 }]
  let readback : Nat → PlausibleWitnessDag.Readback EdgeFit := fun walkSteps =>
    let count := min walkSteps edgeCandidateCount
    let seedPair := edgeCandidateAt 0
    let seed := evaluateEdgeFit lut approxGrid seedPair.1 seedPair.2
    let best := Id.run do
      let mut best := seed
      for idx in [0:count] do
        let pair := edgeCandidateAt idx
        best := chooseBetter best (evaluateEdgeFit lut approxGrid pair.1 pair.2)
      pure best
    { value := best, found := true, witnessIdx := best.maxIdx, budgetHit := false }
  let (fit, _, trace) ← PlausibleWitnessDag.resolve
    "left-edge residual correction"
    (fun _ idx => idx < edgeCandidateCount)
    readback
    levels
  pure (fit, trace)

def parseGroundTruth (s : String) : Except String (Array Float) := do
  let json ← Lean.Json.parse s
  let arr ← json.getArr?
  arr.foldlM (init := #[]) fun acc value => do
    let n ← value.getNum?
    Except.ok (acc.push n.toFloat)

/-- Map a value in [0,1] to an RGB false-colour using a blue→cyan→green→yellow→red gradient. -/
def falseColor (t : Float) : (Nat × Nat × Nat) :=
  let t := max 0.0 (min 1.0 t)
  if t < 0.25 then
    let s := t / 0.25
    (0, (s * 255.0).toUInt8.toNat, 255)
  else if t < 0.5 then
    let s := (t - 0.25) / 0.25
    (0, 255, ((1.0 - s) * 255.0).toUInt8.toNat)
  else if t < 0.75 then
    let s := (t - 0.5) / 0.25
    ((s * 255.0).toUInt8.toNat, 255, 0)
  else
    let s := (t - 0.75) / 0.25
    (255, ((1.0 - s) * 255.0).toUInt8.toNat, 0)

def writePpm (path : System.FilePath) (pixels : Array (Nat × Nat × Nat)) (w h : Nat) : IO Unit := do
  let mut lines : Array String := #[s!"P3\n{w} {h}\n255"]
  for row in [0:h] do
    let mut rowStr := ""
    for col in [0:w] do
      -- Image row 0 = roughness 127 (top = high roughness), col = cosTheta
      let r := h - 1 - row
      let idx := r * w + col
      let (pr, pg, pb) := pixels[idx]!
      rowStr := rowStr ++ s!"{pr} {pg} {pb}  "
    lines := lines.push rowStr
  IO.FS.writeFile path (String.intercalate "\n" lines.toList ++ "\n")

def renderFalseColor (lut approxGrid : Array Float) (fit : EdgeFit) : IO Unit := do
  let approxEC : Array Float := Id.run do
    let mut arr := Array.mkEmpty (128 * 128)
    for i in [0:128] do
      let roughness := gridFloats[i]!
      for j in [0:128] do
        let idx := i * 128 + j
        let cosTheta := gridFloats[j]!
        let y := max (approxGrid[idx]! +
          fit.amplitude * edgeWeight fit.params roughness cosTheta) 0.0
        arr := arr.push y
    pure arr
  let mut rawErrors := Array.mkEmpty (128 * 128)
  let mut ecErrors  := Array.mkEmpty (128 * 128)
  for idx in [0:128 * 128] do
    let truth  := lut[idx]!
    let rawErr := Float.abs (truth - approxGrid[idx]!)
    let ecErr  := Float.abs (truth - approxEC[idx]!)
    rawErrors := rawErrors.push rawErr
    ecErrors  := ecErrors.push ecErr
  -- Use 99th-percentile as scale cap so the bulk of the field gets colour
  let sorted     := rawErrors.toList.mergeSort (· ≤ ·) |>.toArray
  let sortedEC   := ecErrors.toList.mergeSort  (· ≤ ·) |>.toArray
  let p99idx     := (rawErrors.size * 99) / 100
  let p99idxEC   := (ecErrors.size * 99) / 100
  let scale   := max (sorted[p99idx]!)   0.001
  let ecScale := max (sortedEC[p99idxEC]!) 0.001
  let rawPixels := rawErrors.map (fun e => falseColor (e / scale))
  let ecPixels  := ecErrors.map  (fun e => falseColor (e / ecScale))
  IO.FS.createDirAll "rendered"
  writePpm "rendered/comparison_difference_false_color.ppm" rawPixels 128 128
  writePpm "rendered/edge_corrected_difference_false_color.ppm" ecPixels 128 128
  IO.println "wrote rendered/comparison_difference_false_color.ppm"
  IO.println "wrote rendered/edge_corrected_difference_false_color.ppm"
  let cvt ← IO.Process.run { cmd := "which", args := #["convert"] } |>.toBaseIO
  match cvt with
  | .ok _ =>
    let cvtArgs1 : IO.Process.SpawnArgs := {
      cmd := "convert",
      args := #["rendered/comparison_difference_false_color.ppm",
                "rendered/comparison_difference_false_color.png"] }
    let cvtArgs2 : IO.Process.SpawnArgs := {
      cmd := "convert",
      args := #["rendered/edge_corrected_difference_false_color.ppm",
                "rendered/edge_corrected_difference_false_color.png"] }
    _ ← IO.Process.run cvtArgs1
    _ ← IO.Process.run cvtArgs2
    IO.println "wrote rendered/comparison_difference_false_color.png"
    IO.println "wrote rendered/edge_corrected_difference_false_color.png"
  | .error _ =>
    IO.println "(convert not found; PNG conversion skipped)"

def checkGroundTruth : IO Unit := do
  let raw ← IO.FS.readFile lutPath
  let lut ← match parseGroundTruth raw with
    | .ok xs => pure xs
    | .error e => throw <| IO.userError e
  if lut.size != 128 * 128 then
    throw <| IO.userError s!"expected 16384 LUT entries, got {lut.size}"
  let approxGrid := buildApproxGrid
  let mut mse := 0.0
  let mut maxErr := 0.0
  let mut maxIdx := 0
  for i in [0:128] do
    for j in [0:128] do
      let idx := i * 128 + j
      let y := approxGrid[idx]!
      let truth := lut[idx]!
      let err := Float.abs (truth - y)
      mse := mse + err * err
      if err > maxErr then
        maxErr := err
        maxIdx := idx
  mse := mse / 16384.0
  let (fit, trace) ← fitEdgeCorrection lut approxGrid
  IO.println s!"ground truth file: {lutPath}"
  IO.println s!"rank components: {components.length}"
  IO.println s!"degree per factor: {degree}"
  IO.println s!"coefficient count: {coefficientCount}"
  IO.println s!"witness DAG nodes: {witnessDag.length}"
  IO.println s!"component coefficient lengths ok: {components.all componentDegreeOk}"
  IO.println s!"DAG acyclic: {witnessDag.all (fun n => n.deps.all (fun d => d < n.idx))}"
  IO.println s!"ground-truth MSE: {mse}"
  IO.println s!"ground-truth max abs error: {maxErr} at row {maxIdx / 128}, col {maxIdx % 128}"
  IO.println s!"edge fit trace: {reprStr trace}"
  IO.println s!"edge correction amplitude: {fit.amplitude}"
  IO.println s!"edge correction rough smoothstep: {fit.params.rough0} -> {fit.params.rough1}"
  IO.println s!"edge correction cos smoothstep: {fit.params.cos0} -> {fit.params.cos1}"
  IO.println s!"edge-corrected MSE: {fit.mse}"
  IO.println s!"edge-corrected max abs error: {fit.maxErr} at row {fit.maxIdx / 128}, col {fit.maxIdx % 128}"
  -- Render false-colour difference images
  renderFalseColor lut approxGrid fit

end SheenLutMobileCheck

def main : IO Unit :=
  SheenLutMobileCheck.checkGroundTruth
