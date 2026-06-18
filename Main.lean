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

def chebEval (coeffs : Array Float) (offset : Nat) (x : Float) : Float :=
  let t0 := 1.0
  let t1 := x
  let acc := coeffs[offset]! + coeffs[offset + 1]! * t1
  let (_, _, acc) := (List.range 9).foldl (fun (t0, t1, acc) n0 =>
    let n := n0 + 2
    let t2 := 2.0 * x * t1 - t0
    (t1, t2, acc + coeffs[offset + n]! * t2)) (t0, t1, acc)
  acc

def approxAt (roughnessIdx cosThetaIdx : Nat) : Float :=
  let r := warpedCoord roughnessIdx
  let v := warpedCoord cosThetaIdx
  let y := (List.range rankCount).foldl (fun acc k =>
    let offset := k * (degree + 1)
    acc + chebEval roughCoeffs offset r * chebEval cosCoeffs offset v) 0.0
  max y 0.0

def parseGroundTruth (s : String) : Except String (Array Float) := do
  let json ← Lean.Json.parse s
  let arr ← json.getArr?
  arr.foldlM (init := #[]) fun acc value => do
    let n ← value.getNum?
    Except.ok (acc.push n.toFloat)

def checkGroundTruth : IO Unit := do
  let raw ← IO.FS.readFile lutPath
  let lut ← match parseGroundTruth raw with
    | .ok xs => pure xs
    | .error e => throw <| IO.userError e
  if lut.size != 128 * 128 then
    throw <| IO.userError s!"expected 16384 LUT entries, got {lut.size}"
  let mut mse := 0.0
  let mut maxErr := 0.0
  let mut maxIdx := 0
  for i in [0:128] do
    for j in [0:128] do
      let idx := i * 128 + j
      let y := approxAt i j
      let truth := lut[idx]!
      let err := Float.abs (truth - y)
      mse := mse + err * err
      if err > maxErr then
        maxErr := err
        maxIdx := idx
  mse := mse / Float.ofNat (128 * 128)
  IO.println s!"ground truth file: {lutPath}"
  IO.println s!"rank components: {components.length}"
  IO.println s!"degree per factor: {degree}"
  IO.println s!"coefficient count: {coefficientCount}"
  IO.println s!"witness DAG nodes: {witnessDag.length}"
  IO.println s!"component coefficient lengths ok: {components.all componentDegreeOk}"
  IO.println s!"DAG acyclic: {witnessDag.all (fun n => n.deps.all (fun d => d < n.idx))}"
  IO.println s!"ground-truth MSE: {mse}"
  IO.println s!"ground-truth max abs error: {maxErr} at row {maxIdx / 128}, col {maxIdx % 128}"

end SheenLutMobileCheck

def main : IO Unit :=
  SheenLutMobileCheck.checkGroundTruth
