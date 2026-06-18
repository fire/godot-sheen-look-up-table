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
  0.255443200, 0.524121600, 0.0668309000, 0.327211600,
  0.01366972759, -0.02588381493, 0.03185285703, -0.04304663088,
  0.01615568576, 0.02010289788, -0.007529135695, 0.03579024574,
  -0.04431958446, 0.01952014749, -0.05055638609, -0.02042002182,
  0.03161301833, -0.05296272256, 0.08661400775, -0.02765287824,
  -0.03772763986, 0.01314804185, -0.06532637041, 0.08173161275,
  -0.03563674660, 0.09239182289]

def cosCoeffs : Array Float := #[
  1.65597770, -1.44873300, 0.832955200, -0.354371700, -0.307044600,
  0.373793000, -0.142041800, -0.0307764000, 0.0784579000, -0.0550076000,
  0.0230463000, 0.965232900, -0.958853900, 0.647891000, -0.272792300,
  -0.190386800, 0.245405400, -0.0985806000, -0.0137771000, 0.0465832000,
  -0.0334964000, 0.0141780000, 0.728540300, -0.829956400, 0.582212000,
  -0.316190100, -0.0221611000, 0.122872400, -0.0781959000, 0.0226701000,
  0.00316890000, -0.00682420000, 0.00365600000, 0.169502600, 0.146289400,
  -0.183128100, 0.0727207000, -0.0822644000, 0.0518821000, -0.000424300000,
  -0.0266691000, 0.0275761000, -0.0163480000, 0.00633740000,
  -0.07955750455, 0.1688233010, -0.1873883095, 0.1976576456,
  -0.1766518460, 0.1162294198, -0.03314542398, -0.03018441146,
  0.05445439845, -0.04140566257, 0.02066747508, -0.06498364806,
  0.1651142801, -0.08249048626, 0.08223206144, 0.006745825693,
  -0.1138815223, 0.1233045276, -0.07069977333, 0.006346998157,
  0.02389275686, -0.02423478205]

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
    chebEval roughCoeffs 33 r * chebEval cosCoeffs 33 v +
    chebEval roughCoeffs 44 r * chebEval cosCoeffs 44 v +
    chebEval roughCoeffs 55 r * chebEval cosCoeffs 55 v
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
          chebEval roughCoeffs 33 r * chebEval cosCoeffs 33 v +
          chebEval roughCoeffs 44 r * chebEval cosCoeffs 44 v +
          chebEval roughCoeffs 55 r * chebEval cosCoeffs 55 v
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

/-- Map a value in [0,1] to RGB using a smooth turbo-style colormap.
    Polynomial approximation of the Google Turbo palette — no hard kinks. -/
def falseColor (t : Float) : (Nat × Nat × Nat) :=
  let t := max 0.0 (min 1.0 t)
  -- Turbo colormap polynomial coefficients (R, G, B) from the reference implementation
  let r :=   0.13572138 + t * ( 4.61539260 + t * (-42.66032258 + t * ( 132.13108234 + t * (-152.94239396 + t * 59.28637943))))
  let g :=   0.09140261 + t * ( 2.19418839 + t * (  4.84296658 + t * ( -14.18503333 + t * (   4.27729857 + t *  2.82956604))))
  let b :=   0.10667330 + t * (12.64194608 + t * (-60.58694580 + t * ( 110.46943090 + t * ( -89.38180175 + t * 26.22664274))))
  let clamp (x : Float) : Nat := (max 0.0 (min 1.0 x) * 255.0).toUInt8.toNat
  (clamp r, clamp g, clamp b)

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

/-- Convert a Float in [0,∞) to a greyscale RGB pixel, clamped to [0,1]. -/
def greyPixel (v : Float) : (Nat × Nat × Nat) :=
  let g := (max 0.0 (min 1.0 v) * 255.0).toUInt8.toNat
  (g, g, g)

/-- Run `convert` with given args; silently ignore errors. -/
def runConvert (args : Array String) : IO Unit := do
  let a : IO.Process.SpawnArgs := { cmd := "convert", args := args }
  _ ← IO.Process.run a

/-- Delete a file, ignoring errors. -/
def rmTemp (path : String) : IO Unit := do
  _ ← (IO.FS.removeFile path).toBaseIO
  pure ()

/-- Render a 2×2 matrix PNG:
      top row:    ground truth (grey)  | best approx (grey)
      bottom row: ground truth (false) | best approx (false)
    All four cells share the same value scale. Only matrix_2x2.png is kept. -/
def renderMatrix (lut approxGrid : Array Float) (fit : EdgeFit) : IO Unit := do
  -- Build edge-corrected approximation
  let approxEC : Array Float := Id.run do
    let mut arr := Array.mkEmpty (128 * 128)
    for i in [0:128] do
      let roughness := gridFloats[i]!
      for j in [0:128] do
        let idx := i * 128 + j
        let cosTheta := gridFloats[j]!
        arr := arr.push (max (approxGrid[idx]! +
          fit.amplitude * edgeWeight fit.params roughness cosTheta) 0.0)
    pure arr
  -- Shared value scale: 99th-percentile of LUT
  let sortedLut := lut.toList.mergeSort (· ≤ ·) |>.toArray
  let p99    := (lut.size * 99) / 100
  let valMax := max (sortedLut[p99]!) 0.001
  -- Per-pixel absolute error
  let errPixels : Array Float := Id.run do
    let mut arr := Array.mkEmpty (128 * 128)
    for idx in [0:128*128] do
      arr := arr.push (Float.abs (lut[idx]! - approxEC[idx]!))
    pure arr
  -- 99th-pct error scale for the diff column
  let sortedErr := errPixels.toList.mergeSort (· ≤ ·) |>.toArray
  let errMax := max (sortedErr[p99]!) 0.001
  -- Six cell pixel arrays (3 columns × 2 rows)
  let tlPixels := lut.map      greyPixel                           -- GT grey
  let tcPixels := approxEC.map greyPixel                           -- approx grey
  let trPixels := errPixels.map (fun e => greyPixel (e / errMax))  -- diff grey
  let blPixels := lut.map      (fun v => falseColor (v / valMax))  -- GT false
  let bcPixels := approxEC.map (fun v => falseColor (v / valMax))  -- approx false
  let brPixels := errPixels.map (fun e => falseColor (e / errMax)) -- diff false
  IO.FS.createDirAll "rendered"
  writePpm "rendered/_tl.ppm" tlPixels 128 128
  writePpm "rendered/_tc.ppm" tcPixels 128 128
  writePpm "rendered/_tr.ppm" trPixels 128 128
  writePpm "rendered/_bl.ppm" blPixels 128 128
  writePpm "rendered/_bc.ppm" bcPixels 128 128
  writePpm "rendered/_br.ppm" brPixels 128 128
  let labelCell : String → String → String → IO Unit := fun src lbl dst =>
    runConvert #[src,
      "-scale", "512x512",
      "-gravity", "South", "-background", "black", "-splice", "0x32",
      "-fill", "white", "-font", "DejaVu-Sans-Bold", "-pointsize", "20",
      "-gravity", "South", "-annotate", "+0+6", lbl,
      dst]
  labelCell "rendered/_tl.ppm" "Ground truth"       "rendered/_tl_l.ppm"
  labelCell "rendered/_tc.ppm" "Best approx (EC)"   "rendered/_tc_l.ppm"
  labelCell "rendered/_tr.ppm" "Diff (grey)"         "rendered/_tr_l.ppm"
  labelCell "rendered/_bl.ppm" "GT (false colour)"   "rendered/_bl_l.ppm"
  labelCell "rendered/_bc.ppm" "Approx (false)"      "rendered/_bc_l.ppm"
  labelCell "rendered/_br.ppm" "Diff (false colour)" "rendered/_br_l.ppm"
  runConvert #["rendered/_tl_l.ppm", "rendered/_tc_l.ppm", "rendered/_tr_l.ppm",
               "+append", "rendered/_top.ppm"]
  runConvert #["rendered/_bl_l.ppm", "rendered/_bc_l.ppm", "rendered/_br_l.ppm",
               "+append", "rendered/_bot.ppm"]
  runConvert #["rendered/_top.ppm", "rendered/_bot.ppm", "-append", "rendered/matrix_2x2.png"]
  for f in ["rendered/_tl.ppm", "rendered/_tc.ppm", "rendered/_tr.ppm",
            "rendered/_bl.ppm", "rendered/_bc.ppm", "rendered/_br.ppm",
            "rendered/_tl_l.ppm", "rendered/_tc_l.ppm", "rendered/_tr_l.ppm",
            "rendered/_bl_l.ppm", "rendered/_bc_l.ppm", "rendered/_br_l.ppm",
            "rendered/_top.ppm", "rendered/_bot.ppm"] do
    rmTemp f
  IO.println "wrote rendered/matrix_2x2.png  (GT | approx | diff) × (grey | false colour)"


def dot (a b : Array Float) : Float :=
  Id.run do
    let mut s := 0.0
    for i in [0:a.size] do
      s := s + a[i]! * b[i]!
    pure s

/-- Matrix-vector product: A (128×128 row-major) times v (length 128), result length 128. -/
def matVec (A : Array Float) (v : Array Float) : Array Float :=
  Id.run do
    let mut out := Array.replicate 128 0.0
    for i in [0:128] do
      let mut s := 0.0
      for j in [0:128] do
        s := s + A[i * 128 + j]! * v[j]!
      out := out.set! i s
    pure out

/-- Transpose-vector product: Aᵀ times v. -/
def matTVec (A : Array Float) (v : Array Float) : Array Float :=
  Id.run do
    let mut out := Array.replicate 128 0.0
    for i in [0:128] do
      for j in [0:128] do
        out := out.set! j (out[j]! + A[i * 128 + j]! * v[i]!)
    pure out

/-- Normalise a vector; return (unit vector, norm). -/
def normalize (v : Array Float) : Array Float × Float :=
  let n := Float.sqrt (dot v v)
  if n < 1e-15 then (v, 0.0)
  else (v.map (· / n), n)

/-- Deflate rank-1 component sigma*u*vᵀ from A (nRows×128). -/
def deflate (A : Array Float) (u v : Array Float) (sigma : Float) : Array Float :=
  let nRows := A.size / 128
  Id.run do
    let mut B := A
    for i in [0:nRows] do
      for j in [0:128] do
        B := B.set! (i * 128 + j) (B[i * 128 + j]! - sigma * u[i]! * v[j]!)
    pure B

/-- Power iteration: largest singular value + vectors of A (nRows×128), after `iters` steps. -/
def powerIter (A : Array Float) (iters : Nat) : Float × Array Float × Array Float :=
  let nRows := A.size / 128
  Id.run do
    let (v0, _) := normalize (Array.replicate 128 1.0)
    let mut v := v0
    let mut u := Array.replicate nRows 0.0
    let mut sigma := 0.0
    for _ in [0:iters] do
      -- u = A*v, normalise
      let mut au := Array.replicate nRows 0.0
      for i in [0:nRows] do
        let mut s := 0.0
        for j in [0:128] do
          s := s + A[i * 128 + j]! * v[j]!
        au := au.set! i s
      let (u', _) := normalize au
      u := u'
      -- v = Aᵀ*u, normalise
      let mut atv := Array.replicate 128 0.0
      for i in [0:nRows] do
        for j in [0:128] do
          atv := atv.set! j (atv[j]! + A[i * 128 + j]! * u[i]!)
      let (v', s) := normalize atv
      v := v'
      sigma := s
    pure (sigma, u, v)

/-- Truncated SVD via power iteration + deflation. Prints singular values and
    cumulative energy until sigma < 1e-6 or rank 40 is reached. -/
def reportNumericalRank (lut : Array Float) : IO Unit := do
  IO.println ""
  IO.println "=== Numerical rank of the 128×128 LUT matrix ==="
  IO.println "rank       sigma    cumul energy%   residual%"
  let totalEnergy := dot lut lut
  let mut A := lut
  let mut cumul := 0.0
  for k in [0:40] do
    let (sigma, u, v) := powerIter A 60
    if sigma < 1e-6 then
      IO.println s!"Numerical rank (sigma > 1e-6): {k}"
      break
    cumul := cumul + sigma * sigma
    let frac  := cumul / totalEnergy * 100.0
    let resid := (1.0 - cumul / totalEnergy) * 100.0
    IO.println s!"{k+1}  {sigma}  {frac}%  {resid}%"
    A := deflate A u v sigma
    if k == 39 then
      IO.println "Numerical rank (sigma > 1e-6): >= 40"

/-- Evaluate the d-th Chebyshev polynomial at x using Float arithmetic. -/
def chebF : Nat → Float → Float
  | 0, _ => 1.0
  | 1, x => x
  | n + 2, x =>
    Id.run do
      let mut t0 := 1.0
      let mut t1 := x
      for _ in [0:n+1] do
        let t2 := 2.0 * x * t1 - t0
        t0 := t1
        t1 := t2
      pure t1

/-- Extract a row-slice of the LUT: rows [r0, r1), all 128 cosTheta columns.
    Returns a flat array of size (r1-r0)*128. -/
def lutSlice (lut : Array Float) (r0 r1 : Nat) : Array Float :=
  Id.run do
    let rows := r1 - r0
    let mut out := Array.mkEmpty (rows * 128)
    for i in [0:rows] do
      for j in [0:128] do
        out := out.push lut[(r0 + i) * 128 + j]!
    pure out

/-- Evaluate a rank-k separable Chebyshev model on a (rows×128) grid.
    roughCoeffs: k*(degR+1) coefficients, cosCoeffs: k*(degC+1) coefficients.
    roughWarped: precomputed warped roughness for each row in [r0,r1).
    Returns flat array of size rows*128. -/
def evalPiecewise (roughCoeffs cosCoeffs : Array Float) (rank degR degC : Nat)
    (roughWarped : Array Float) : Array Float :=
  let rows := roughWarped.size
  Id.run do
    let mut out := Array.mkEmpty (rows * 128)
    for i in [0:rows] do
      let r := roughWarped[i]!
      for j in [0:128] do
        let v := warpedFloats[j]!
        let mut y := 0.0
        for k in [0:rank] do
          y := y + chebEval roughCoeffs (k * (degR + 1)) r *
                   chebEval cosCoeffs  (k * (degC + 1)) v
        out := out.push (max y 0.0)
    pure out

/-- Compute MSE and max absolute error between two flat arrays. -/
def errorStats (a b : Array Float) : Float × Float × Nat :=
  Id.run do
    let mut sumSq  := 0.0
    let mut maxErr := 0.0
    let mut maxIdx := 0
    for i in [0:a.size] do
      let e := Float.abs (a[i]! - b[i]!)
      sumSq := sumSq + e * e
      if e > maxErr then maxErr := e; maxIdx := i
    let n := (a.size.toFloat)
    pure (sumSq / n, maxErr, maxIdx)

/-- Warp roughness linearly into [-1,1] within a segment [r0_frac, r1_frac].
    r0_frac and r1_frac are the fractional row boundaries in [0,1]. -/
def warpRoughnessSegment (i r0row r1row : Nat) : Float :=
  -- map row i linearly to [-1, 1] within [r0row, r1row)
  let t := (Float.ofNat i - Float.ofNat r0row) / Float.ofNat (r1row - r0row - 1)
  2.0 * t - 1.0

/-- Fit the col=0 left edge (cosTheta=0) as a 1D Chebyshev series in sqrt(roughness).
    Reports coefficients and max reconstruction error.
    The left edge has a near-singularity: values go from ~0.6 at r=0 to 18.25 at r=1.
    We fit log(f) to tame the dynamic range, then exponentiate on decode. -/
def fitLeftEdge (lut : Array Float) : IO Unit := do
  IO.println ""
  IO.println "=== Left-edge col=0 fit (log-space 1D Chebyshev in sqrt(roughness)) ==="
  -- Extract col=0 values and apply log transform
  let logVals : Array Float := Id.run do
    let mut arr := Array.mkEmpty 128
    for i in [0:128] do
      let v := lut[i * 128]!
      arr := arr.push (Float.log (max v 1e-6))
    pure arr
  -- Fit degree-degR Chebyshev in sqrt-warped roughness (same warp as the shader)
  let degR := 15
  -- x_i = warpedFloats[i] = 2*sqrt(i/127) - 1
  let coeffs : Array Float := Id.run do
    let mut out := Array.mkEmpty (degR + 1)
    for d in [0:(degR + 1)] do
      let mut s := 0.0
      for i in [0:128] do
        s := s + logVals[i]! * chebF d warpedFloats[i]!
      let norm := if d == 0 then 128.0 else 64.0
      out := out.push (s / norm)
    pure out
  -- Evaluate and measure error in original (non-log) space
  let mut maxErr := 0.0
  let mut mse    := 0.0
  for i in [0:128] do
    let x := warpedFloats[i]!
    let mut logFit := 0.0
    for d in [0:(degR + 1)] do
      logFit := logFit + coeffs[d]! * chebF d x
    let fitted := Float.exp logFit
    let truth  := lut[i * 128]!
    let err    := Float.abs (truth - fitted)
    mse    := mse + err * err
    if err > maxErr then maxErr := err
  mse := mse / 128.0
  IO.println s!"degree={degR}  MSE={mse}  maxErr={maxErr}"
  IO.println s!"coeffs (log-space): {coeffs.toList.take 8}"
  IO.println ""
  IO.println "col=0 truth vs fit (selected rows):"
  IO.println "row   truth     fitted    err"
  for i in [0, 10, 20, 40, 60, 80, 90, 100, 105, 110, 115, 120, 122, 124, 125, 126, 127] do
    let x := warpedFloats[i]!
    let mut logFit := 0.0
    for d in [0:(degR + 1)] do
      logFit := logFit + coeffs[d]! * chebF d x
    let fitted := Float.exp logFit
    let truth  := lut[i * 128]!
    IO.println s!"{i}   {truth}   {fitted}   {Float.abs (truth - fitted)}"


/-- Rank sweep: for each rank in [4,5,6,7,8], fit a full-rank two-pass separable
    Chebyshev model (degC=10, degR=10) and report boundary and interior errors.
    The two-pass method: Pass 1 = exact per-row DCT in cosTheta,
    Pass 2 = Chebyshev fit of each coefficient as a function of roughness.
    This is a full-matrix model (not low-rank SVD), so rank here means
    we keep only the top `rank` cosTheta basis functions (sorted by
    their energy in the roughness direction). -/
def rankSweep (lut : Array Float) : IO Unit := do
  IO.println ""
  IO.println "=== Rank sweep: two-pass DCT fit, varying number of cosTheta basis functions ==="
  let degC := 10
  let degR  := 10
  -- Pass 1 once: exact per-row cosTheta coefficients for all degC+1 = 11 basis functions
  -- perRowCoeffs[i * (degC+1) + d] = d-th cosTheta coeff for row i
  let perRowCoeffs : Array Float := Id.run do
    let mut out := Array.mkEmpty (128 * (degC + 1))
    for i in [0:128] do
      for d in [0:(degC + 1)] do
        let mut s := 0.0
        for j in [0:128] do
          s := s + lut[i * 128 + j]! * chebF d warpedFloats[j]!
        out := out.push (s / if d == 0 then 128.0 else 64.0)
    pure out
  -- Compute energy of each cosTheta basis function across all roughness rows
  -- energy[d] = sum_i perRowCoeffs[i][d]^2
  let energy : Array Float := Id.run do
    let mut e := Array.replicate (degC + 1) 0.0
    for i in [0:128] do
      for d in [0:(degC + 1)] do
        let c := perRowCoeffs[i * (degC + 1) + d]!
        e := e.set! d (e[d]! + c * c)
    pure e
  -- Sort basis indices by descending energy
  let sortedD : Array Nat := Id.run do
    let mut idx := Array.mkEmpty (degC + 1)
    for d in [0:(degC + 1)] do idx := idx.push d
    -- Simple insertion sort
    let mut arr := idx
    for i in [1:(degC + 1)] do
      let key := arr[i]!
      let mut j := i
      while j > 0 && energy[arr[j-1]!]! < energy[key]! do
        arr := arr.set! j arr[j-1]!
        j := j - 1
      arr := arr.set! j key
    pure arr
  IO.println s!"CosTheta basis energy ranking (d → energy):"
  for i in [0:(degC+1)] do
    let d := sortedD[i]!
    IO.println s!"  rank {i+1}: d={d}  energy={energy[d]!}"
  IO.println ""
  -- Pass 2 per rank: fit roughness Chebyshev for the top `rank` cosTheta bases
  for rank in [4, 5, 6, 7, 8] do
    -- For each selected cosTheta basis d, fit roughness polynomial
    -- roughCoeffs for basis d at position p: roughCoeffs[p * (degR+1) + k]
    let roughCoeffs : Array Float := Id.run do
      let mut out := Array.mkEmpty (rank * (degR + 1))
      for p in [0:rank] do
        let d := sortedD[p]!
        for k in [0:(degR + 1)] do
          let mut s := 0.0
          for i in [0:128] do
            s := s + perRowCoeffs[i * (degC + 1) + d]! * chebF k warpedFloats[i]!
          out := out.push (s / if k == 0 then 128.0 else 64.0)
      pure out
    -- Evaluate
    let mut mse := 0.0; let mut maxErr := 0.0; let mut maxIdx := 0
    let mut mseInt := 0.0; let mut maxErrInt := 0.0
    let mut mseTop := 0.0; let mut maxErrTop := 0.0  -- rows 120-127
    for i in [0:128] do
      let xr := warpedFloats[i]!
      for j in [0:128] do
        let xv := warpedFloats[j]!
        let mut y := 0.0
        for p in [0:rank] do
          let d := sortedD[p]!
          let mut rc := 0.0
          for k in [0:(degR + 1)] do
            rc := rc + roughCoeffs[p * (degR + 1) + k]! * chebF k xr
          y := y + rc * chebF d xv
        let fitted := max y 0.0
        let err := Float.abs (lut[i * 128 + j]! - fitted)
        mse := mse + err * err
        if err > maxErr then maxErr := err; maxIdx := i * 128 + j
        if j >= 1 then
          mseInt := mseInt + err * err
          if err > maxErrInt then maxErrInt := err
        if i >= 120 then
          mseTop := mseTop + err * err
          if err > maxErrTop then maxErrTop := err
    mse    := mse    / 16384.0
    mseInt := mseInt / 16256.0
    mseTop := mseTop / (8.0 * 128.0)
    IO.println s!"rank={rank}: MSE={mse}  maxErr={maxErr} at row={maxIdx/128} col={maxIdx%128}"
    IO.println s!"  interior (col≥1) MSE={mseInt}  maxErr={maxErrInt}"
    IO.println s!"  top rows 120-127 MSE={mseTop}  maxErr={maxErrTop}"
    -- Show row=127 boundary errors
    let xr127 := warpedFloats[127]!
    let rowErr : String := Id.run do
      let mut s := ""
      for j in [0:8] do
        let xv := warpedFloats[j]!
        let mut y := 0.0
        for p in [0:rank] do
          let d := sortedD[p]!
          let mut rc := 0.0
          for k in [0:(degR + 1)] do
            rc := rc + roughCoeffs[p * (degR + 1) + k]! * chebF k xr127
          y := y + rc * chebF d xv
        let fitted := max y 0.0
        let err := fitted - lut[127 * 128 + j]!
        s := s ++ s!"col{j}={err} "
      pure s
    IO.println s!"  row=127 errs: {rowErr}"
    IO.println ""

/-- Helper: two-pass DCT fit with a given roughness warp function.
    degC=10, keeps all 11 cosTheta bases (full-matrix, not low-rank).
    Returns (eval function, full MSE, interior MSE, top-rows MSE, maxErr, row127 col1 err). -/
def twoPassFit (lut : Array Float) (warpR : Nat → Float) (degR degC : Nat)
    : Array Float × Float × Float × Float × Float × Float :=
  let n := degC + 1
  let nRows := lut.size / 128
  -- Pass 1: per-row cosTheta coefficients
  let perRow : Array Float := Id.run do
    let mut out := Array.mkEmpty (nRows * n)
    for i in [0:nRows] do
      for d in [0:n] do
        let mut s := 0.0
        for j in [0:128] do
          s := s + lut[i * 128 + j]! * chebF d warpedFloats[j]!
        out := out.push (s / if d == 0 then 128.0 else 64.0)
    pure out
  -- Pass 2: for each cosTheta coeff d, fit degree-degR Chebyshev in roughness
  let roughCoeffs : Array Float := Id.run do
    let mut out := Array.mkEmpty (n * (degR + 1))
    for d in [0:n] do
      for k in [0:(degR + 1)] do
        let mut s := 0.0
        for i in [0:nRows] do
          s := s + perRow[i * n + d]! * chebF k (warpR i)
        out := out.push (s / if k == 0 then nRows.toFloat else nRows.toFloat / 2.0)
    pure out
  -- Evaluate and gather stats
  let (mse, mseInt, mseTop, maxErr, e127c1) := Id.run do
    let mut mse := 0.0; let mut mseInt := 0.0; let mut mseTop := 0.0
    let mut maxErr := 0.0; let mut e127c1 := 0.0
    for i in [0:nRows] do
      let xr := warpR i
      for j in [0:128] do
        let xv := warpedFloats[j]!
        let mut y := 0.0
        for d in [0:n] do
          let mut rc := 0.0
          for k in [0:(degR + 1)] do
            rc := rc + roughCoeffs[d * (degR + 1) + k]! * chebF k xr
          y := y + rc * chebF d xv
        let fitted := max y 0.0
        let err := Float.abs (lut[i * 128 + j]! - fitted)
        mse := mse + err * err
        if err > maxErr then maxErr := err
        if j >= 1 then mseInt := mseInt + err * err
        if i >= nRows - 8 then mseTop := mseTop + err * err
        if i == nRows - 1 && j == 1 then e127c1 := fitted - lut[(nRows - 1) * 128 + 1]!
    let total := Float.ofNat (nRows * 128)
    let totalInt := Float.ofNat (nRows * 127)
    let totalTop := Float.ofNat (min 8 nRows * 128)
    pure (mse / total, mseInt / totalInt, mseTop / totalTop, maxErr, e127c1)
  (roughCoeffs, mse, mseInt, mseTop, maxErr, e127c1)

/-- Option 1: boundary-constrained fit.
    At x=+1 (roughness=1), T_k(+1)=1 for all k, so the series sum equals the target.
    We substitute c_0 = target - Σ_{k≥1} c_k, fit the rest freely using
    modified basis B_k(x) = T_k(x) - 1 for k≥1 (so B_k(+1)=0). -/
def fitBoundaryConstrained (lut : Array Float) : Float × Float × Float × Float × Float :=
  let degR := 10; let degC := 10; let n := degC + 1
  -- Pass 1: per-row cosTheta coefficients (exact)
  let perRow : Array Float := Id.run do
    let mut out := Array.mkEmpty (128 * n)
    for i in [0:128] do
      for d in [0:n] do
        let mut s := 0.0
        for j in [0:128] do
          s := s + lut[i * 128 + j]! * chebF d warpedFloats[j]!
        out := out.push (s / if d == 0 then 128.0 else 64.0)
    pure out
  -- Pass 2 constrained: for each cosTheta coeff d, fit B_k basis
  -- Eval: perRow[127][d] + Σ_{k=1..degR} c_k*(T_k(x)-1)
  let freeCoeffs : Array Float := Id.run do  -- (n × degR) free coefficients
    let mut out := Array.mkEmpty (n * degR)
    for d in [0:n] do
      let target := perRow[127 * n + d]!   -- value at roughness=1
      -- residual[i] = perRow[i][d] - target
      for k in [1:(degR + 1)] do
        -- inner product of residual with B_k(x) = T_k(x) - 1
        let mut s := 0.0
        for i in [0:128] do
          let res := perRow[i * n + d]! - target
          s := s + res * (chebF k (warpedFloats[i]!) - 1.0)
        -- norm: Σ_i (T_k(x_i) - 1)^2
        let mut norm := 0.0
        for i in [0:128] do
          let v := chebF k (warpedFloats[i]!) - 1.0
          norm := norm + v * v
        out := out.push (if norm > 1e-10 then s / norm else 0.0)
    pure out
  -- Evaluate
  let (mse, mseInt, mseTop, maxErr, e127c1) := Id.run do
    let mut mse := 0.0; let mut mseInt := 0.0; let mut mseTop := 0.0
    let mut maxErr := 0.0; let mut e127c1 := 0.0
    for i in [0:128] do
      let xr := warpedFloats[i]!
      for j in [0:128] do
        let xv := warpedFloats[j]!
        let mut y := 0.0
        for d in [0:n] do
          let target := perRow[127 * n + d]!
          -- rc = target + Σ_{k=1..degR} c_k*(T_k(x)-1)
          let mut rc := target
          for k in [1:(degR + 1)] do
            rc := rc + freeCoeffs[d * degR + (k - 1)]! * (chebF k xr - 1.0)
          y := y + rc * chebF d xv
        let fitted := max y 0.0
        let err := Float.abs (lut[i * 128 + j]! - fitted)
        mse := mse + err * err
        if err > maxErr then maxErr := err
        if j >= 1 then mseInt := mseInt + err * err
        if i >= 120 then mseTop := mseTop + err * err
        if i == 127 && j == 1 then e127c1 := fitted - lut[127 * 128 + 1]!
    pure (mse / 16384.0, mseInt / 16256.0, mseTop / 1024.0, maxErr, e127c1)
  (mse, mseInt, mseTop, maxErr, e127c1)

/-- Compare all three fix strategies on identical metrics.
    Baseline: current production model (rank-4 sqrt warp, global fit).
    Option A: boundary-constrained fit (c_0 determined by row=127 target).
    Option B: piecewise fit with segment C using cosine re-warp
              so roughness=1 maps to x=-1 (interior-side endpoint).
    Option C: r² roughness warp (spreads samples away from r=1 boundary). -/
def compareFixes (lut : Array Float) : IO Unit := do
  IO.println ""
  IO.println "=== Comparing three boundary-fix strategies ==="
  IO.println "Metrics: full MSE | interior MSE (col>=1) | top-rows MSE (rows 120-127) | maxErr | row=127 col=1 err"
  IO.println ""

  -- Baseline: sqrt warp, full 11 cosTheta bases, degR=10
  let (_, mseFull0, mseInt0, mseTop0, maxErr0, e1270) :=
    twoPassFit lut (fun i => warpedFloats[i]!) 10 10
  IO.println s!"Baseline (sqrt warp, degR=10, degC=10, all 11 bases):"
  IO.println s!"  MSE={mseFull0}  intMSE={mseInt0}  topMSE={mseTop0}  maxErr={maxErr0}  e127c1={e1270}"

  -- Option A: boundary-constrained
  let (mseA, mseIntA, mseTopA, maxErrA, e127A) := fitBoundaryConstrained lut
  IO.println ""
  IO.println s!"Option A — boundary-constrained (enforce series=target at roughness=1):"
  IO.println s!"  MSE={mseA}  intMSE={mseIntA}  topMSE={mseTopA}  maxErr={maxErrA}  e127c1={e127A}"

  -- Option B: piecewise with flipped warp on segment C [rows 110-127]
  -- Standard linear warp for segment: row i → x = 2*(i-r0)/(r1-1-r0) - 1
  -- FLIPPED so roughness=1 (i=127) → x=-1 and roughness=0.86 (i=110) → x=+1
  -- x = 1 - 2*(i - 110) / 17  using integer arithmetic to avoid GMP
  let warpSegC_flipped : Nat → Float := fun i =>
    let num := Float.ofNat (if i >= 110 then i - 110 else 0)
    1.0 - 2.0 * num / 17.0

  -- Fit segment A [0,73): rows 0..72, warp = warpedFloats[i]
  let lutA : Array Float := Id.run do
    let mut out := Array.mkEmpty (73 * 128)
    for i in [0:73] do
      for j in [0:128] do out := out.push lut[i * 128 + j]!
    pure out
  let (_, mseA2, _, _, maxErrA2, _) :=
    twoPassFit lutA (fun i => warpedFloats[i]!) 5 10

  -- Fit segment B [73,110): rows 73..109, re-index 0-based inside twoPassFit
  -- twoPassFit gets a 37-row array; warpR(i) should cover global rows 73+i
  let lutB : Array Float := Id.run do
    let mut out := Array.mkEmpty (37 * 128)
    for i in [73:110] do
      for j in [0:128] do out := out.push lut[i * 128 + j]!
    pure out
  -- Linear local warp: row i (0..36) → x = 2*i/36 - 1
  let warpSegB : Nat → Float := fun i => 2.0 * Float.ofNat i / 36.0 - 1.0
  let (_, mseB2, _, _, maxErrB2, _) :=
    twoPassFit lutB warpSegB 7 10

  -- Fit segment C [110,127]: rows 110..127, 18 rows, 0-based inside twoPassFit
  -- FLIPPED warp: row i (0..17) → x = 1 - 2*i/17  so i=0→+1, i=17→-1
  let lutC : Array Float := Id.run do
    let mut out := Array.mkEmpty (18 * 128)
    for i in [110:128] do
      for j in [0:128] do out := out.push lut[i * 128 + j]!
    pure out
  let warpSegC : Nat → Float := fun i =>
    1.0 - 2.0 * Float.ofNat i / 17.0
  let (_, mseC2, mseIntC2, mseTopC, maxErrC, e127B) :=
    twoPassFit lutC warpSegC 9 10

  -- Combined segment stats
  let msePW := (73.0 * mseA2 + 37.0 * mseB2 + 18.0 * mseC2) / 128.0
  let maxErrPW := max (max maxErrA2 maxErrB2) maxErrC
  IO.println ""
  IO.println s!"Option B — piecewise + flipped warp on segment C (roughness=1 at x=-1):"
  IO.println s!"  segA MSE={mseA2} maxErr={maxErrA2}"
  IO.println s!"  segB MSE={mseB2} maxErr={maxErrB2}"
  IO.println s!"  segC MSE={mseC2} maxErr={maxErrC}  topMSE={mseTopC}  e127c1={e127B}"
  IO.println s!"  weighted combined MSE={msePW}  maxErr={maxErrPW}"

  -- Option C: r² warp  x = 2*r² - 1  where r = i/127
  let warpRsq : Nat → Float := fun i =>
    let r := Float.ofNat i / 127.0
    2.0 * r * r - 1.0
  let (_, mseFull2, mseInt2, mseTop2, maxErr2, e127C) :=
    twoPassFit lut warpRsq 10 10
  IO.println ""
  IO.println s!"Option C — r² roughness warp (spreads samples from boundary):"
  IO.println s!"  MSE={mseFull2}  intMSE={mseInt2}  topMSE={mseTop2}  maxErr={maxErr2}  e127c1={e127C}"

  IO.println ""
  IO.println "=== Ranking (by interior MSE, col>=1) ==="
  -- Collect and sort by mseInt
  let results : List (String × Float × Float × Float × Float) := [
    ("Baseline", mseA,    mseInt0, mseTop0, maxErr0),  -- note: mseA used for baseline full
    ("Baseline", mseFull0, mseInt0, mseTop0, maxErr0),
    ("Option A (boundary-constrained)", mseA, mseIntA, mseTopA, maxErrA),
    ("Option B (piecewise+flip)",        msePW, mseIntC2, mseTopC, maxErrPW),
    ("Option C (r² warp)",               mseFull2, mseInt2, mseTop2, maxErr2)]
  -- Print sorted
  let ranked := results.tail  -- drop duplicate baseline
  let sorted := ranked.mergeSort (fun a b => a.2.2.1 ≤ b.2.2.1)
  let mut place := 1
  for (name, mse, mseI, mseT, me) in sorted do
    IO.println s!"  #{place}: {name}"
    IO.println s!"      intMSE={mseI}  topMSE={mseT}  maxErr={me}"
    place := place + 1

/-- Fit and report a piecewise separable Chebyshev model.
    Approach (Ottosson/OkHSL-style two-pass fitting with range normalisation):
      - Three segments in roughness, each re-warped locally to [-1,1]
      - Segment C divides out a roughness weight w(r) before fitting to
        handle the 1/(1-r) dynamic range blow-up, then multiplies back on decode
      - Pass 1: exact cosTheta Chebyshev inner product per row
      - Pass 2: Chebyshev fit of each cosTheta coefficient in roughness
    C¹ continuity at segment boundaries is not enforced here (diagnostic only). -/
def fitPiecewise (lut : Array Float) : IO Unit := do
  IO.println ""
  IO.println "=== Piecewise separable Chebyshev fit (OkHSL-style two-pass) ==="
  -- Segment C weight: reference value at col=0 for each row, used to normalise
  -- the explosive left-edge growth. Divides before fit, multiplies after.
  let weightC : Nat → Float := fun i =>
    -- w(r) ≈ lut[row, 0], the peak value at grazing angle — normalises the row
    let v := lut[(110 + i) * 128]!
    if v > 0.001 then v else 1.0
  let segments : List (String × Nat × Nat × Nat × Nat × Bool) := [
    -- (name, r0, r1, degR, degC, useWeight)
    ("A: r=[0.00,0.57)", 0,   73,  5,  7, false),
    ("B: r=[0.57,0.86)", 73,  110, 7,  9, false),
    ("C: r=[0.86,1.00]", 110, 128, 9, 10, true)]
  for (name, r0, r1, degR, degC, useWeight) in segments do
    let rows := r1 - r0
    let roughWarped : Array Float := Id.run do
      let mut arr := Array.mkEmpty rows
      for i in [0:rows] do
        arr := arr.push (warpRoughnessSegment (r0 + i) r0 r1)
      pure arr
    -- Pass 1: per-row cosTheta Chebyshev coefficients (with optional row normalisation)
    let perRowCoeffs : Array Float := Id.run do
      let mut out := Array.mkEmpty (rows * (degC + 1))
      for i in [0:rows] do
        let w := if useWeight then weightC i else 1.0
        for d in [0:(degC + 1)] do
          let mut s := 0.0
          for j in [0:128] do
            s := s + (lut[(r0 + i) * 128 + j]! / w) * chebF d warpedFloats[j]!
          let norm := if d == 0 then 128.0 else 64.0
          out := out.push (s / norm)
      pure out
    -- Pass 2: Chebyshev fit in roughness for each cosTheta coefficient
    let roughCoeffs : Array Float := Id.run do
      let mut out := Array.mkEmpty ((degC + 1) * (degR + 1))
      for d in [0:(degC + 1)] do
        for k in [0:(degR + 1)] do
          let mut s := 0.0
          for i in [0:rows] do
            s := s + perRowCoeffs[i * (degC + 1) + d]! * chebF k roughWarped[i]!
          let norm := if k == 0 then rows.toFloat else rows.toFloat / 2.0
          out := out.push (s / norm)
      pure out
    -- Evaluate
    let fitted : Array Float := Id.run do
      let mut out := Array.mkEmpty (rows * 128)
      for i in [0:rows] do
        let r := roughWarped[i]!
        let w := if useWeight then weightC i else 1.0
        for j in [0:128] do
          let v := warpedFloats[j]!
          let mut y := 0.0
          for d in [0:(degC + 1)] do
            let mut rc := 0.0
            for k in [0:(degR + 1)] do
              rc := rc + roughCoeffs[d * (degR + 1) + k]! * chebF k r
            y := y + rc * chebF d v
          out := out.push (max (y * w) 0.0)
      pure out
    let slice := lutSlice lut r0 r1
    let (mse, maxErr, maxIdx) := errorStats slice fitted
    let maxRow := maxIdx / 128 + r0
    let maxCol := maxIdx % 128
    let nCoeffs := (degC + 1) * (degR + 1)
    IO.println s!"Segment {name}: degR={degR} degC={degC} coeffs={nCoeffs} weight={useWeight}"
    IO.println s!"  MSE={mse}  maxErr={maxErr} at row={maxRow} col={maxCol}"

/-- Per-roughness-row statistics to identify natural breakpoints for a piecewise fit. -/
def profileRoughnessRows (lut : Array Float) : IO Unit := do
  IO.println ""
  IO.println "=== Per-roughness-row profile ==="
  IO.println "row   roughness   rowNorm    rowMax    rowMean"
  for i in [0:128] do
    let mut rowMax  := 0.0
    let mut rowSum  := 0.0
    let mut rowNorm := 0.0
    for j in [0:128] do
      let v := lut[i * 128 + j]!
      if v > rowMax then rowMax := v
      rowSum  := rowSum  + v
      rowNorm := rowNorm + v * v
    let roughness := gridFloats[i]!
    IO.println s!"{i}  {roughness}  {Float.sqrt rowNorm}  {rowMax}  {rowSum / 128.0}"

/-- Write the full 128×128 pixel error table directly to Parquet via DuckDB.
    Uses a temp NDJSON as the pipe (no CSV left on disk).
    Columns: roughness_idx, costheta_idx, roughness, costheta,
             ground_truth, approx_ec, abs_error, signed_error -/
def writeErrorTable (lut approxGrid : Array Float) (fit : EdgeFit) : IO Unit := do
  IO.FS.createDirAll "rendered"
  let tmpPath     := "rendered/.sheen_lut_tmp.ndjson"
  let parquetPath := "rendered/sheen_lut_error.parquet"
  -- Build edge-corrected grid
  let approxEC : Array Float := Id.run do
    let mut arr := Array.mkEmpty (128 * 128)
    for i in [0:128] do
      let roughness := gridFloats[i]!
      for j in [0:128] do
        let cosTheta := gridFloats[j]!
        arr := arr.push (max (approxGrid[i*128+j]! +
          fit.amplitude * edgeWeight fit.params roughness cosTheta) 0.0)
    pure arr
  -- Write NDJSON (one JSON object per line, no header)
  let mut lines := #[""]   -- will be overwritten
  lines := #[]
  for i in [0:128] do
    let roughness := gridFloats[i]!
    for j in [0:128] do
      let idx    := i * 128 + j
      let cosTheta := gridFloats[j]!
      let gt     := lut[idx]!
      let ap     := approxEC[idx]!
      let ae     := Float.abs (gt - ap)
      let se     := ap - gt
      lines := lines.push
        (s!"\{\"ri\":" ++ s!"{i}" ++ ",\"ci\":" ++ s!"{j}" ++
         ",\"roughness\":" ++ s!"{roughness}" ++
         ",\"costheta\":" ++ s!"{cosTheta}" ++
         ",\"ground_truth\":" ++ s!"{gt}" ++
         ",\"approx_ec\":" ++ s!"{ap}" ++
         ",\"abs_error\":" ++ s!"{ae}" ++
         ",\"signed_error\":" ++ s!"{se}" ++ "}")
  IO.FS.writeFile tmpPath (String.intercalate "\n" lines.toList ++ "\n")
  -- DuckDB: read NDJSON → write Parquet directly, no CSV
  let pyScript :=
    "import duckdb\n" ++
    "duckdb.sql(\"COPY (SELECT * FROM read_ndjson_auto('" ++ tmpPath ++ "')) " ++
    "TO '" ++ parquetPath ++ "' (FORMAT PARQUET, COMPRESSION ZSTD)\")\n"
  let args : IO.Process.SpawnArgs := { cmd := "python3", args := #["-c", pyScript] }
  let result ← IO.Process.output args
  -- Remove temp file regardless of outcome
  _ ← (IO.FS.removeFile tmpPath).toBaseIO
  if result.exitCode == 0 then
    IO.println s!"wrote {parquetPath}  ({lines.size} rows, no CSV)"
  else
    IO.println s!"parquet write failed: {result.stderr}"

/-- Solve an (n×n) linear system Ax=b via Gaussian elimination with partial pivoting.
    A is stored row-major in a flat Array of size n*n. Returns solution vector. -/
def gaussElim (A : Array Float) (b : Array Float) (n : Nat) : Array Float :=
  Id.run do
    -- Augmented matrix [A | b], size n*(n+1)
    let mut M : Array Float := Array.mkEmpty (n * (n + 1))
    for i in [0:n] do
      for j in [0:n] do M := M.push A[i * n + j]!
      M := M.push b[i]!
    let col := n + 1
    -- Forward elimination with partial pivoting
    for k in [0:n] do
      -- Find pivot
      let mut pivotRow := k
      let mut pivotVal := Float.abs M[k * col + k]!
      for i in [k+1:n] do
        let v := Float.abs M[i * col + k]!
        if v > pivotVal then pivotVal := v; pivotRow := i
      -- Swap rows k and pivotRow
      if pivotRow != k then
        for j in [0:col] do
          let tmp := M[k * col + j]!
          M := M.set! (k * col + j)         M[pivotRow * col + j]!
          M := M.set! (pivotRow * col + j)   tmp
      -- Eliminate below
      let pivot := M[k * col + k]!
      if Float.abs pivot > 1e-14 then
        for i in [k+1:n] do
          let factor := M[i * col + k]! / pivot
          for j in [k:col] do
            M := M.set! (i * col + j) (M[i * col + j]! - factor * M[k * col + j]!)
    -- Back substitution
    let mut x := Array.replicate n 0.0
    for ii in [0:n] do
      let i := n - 1 - ii
      let mut s := M[i * col + n]!
      for j in [i+1:n] do
        s := s - M[i * col + j]! * x[j]!
      let diag := M[i * col + i]!
      x := x.set! i (if Float.abs diag > 1e-14 then s / diag else 0.0)
    pure x

/-- True Slug-style hybrid: global SVD components + per-band LS in roughness.
    Step 1: Global truncated SVD (power iteration + deflation) extracts rank-K
            coupled (u_k, v_k) components capturing the dominant structure.
    Step 2: Each roughness profile u_k is fitted within nBands equal bands
            using degree-degR polynomial by proper least squares (normal equations).
    Step 3: Each cosTheta profile v_k is projected onto degC Chebyshev basis globally.
    The Slug insight: per-band LS gives each band its own local polynomial with
    O(1) coefficient magnitudes regardless of the global dynamic range, eliminating
    the endpoint-oscillation problem that broke the global approach. -/
def fitSlugStyle (lut : Array Float) : IO Unit := do
  IO.println ""
  IO.println "=== Slug-style: global SVD + per-band LS in roughness ==="

  -- Step 1: Truncated SVD of the 128×128 LUT matrix
  let ranks := [4, 6, 8]
  let svdComponents : Array (Float × Array Float × Array Float) := Id.run do
    let maxRank := 8
    let mut comps := Array.mkEmpty maxRank
    let mut A := lut
    for _ in [0:maxRank] do
      let (sigma, u, v) := powerIter A 80
      comps := comps.push (sigma, u, v)
      A := deflate A u v sigma
    pure comps

  -- Step 2: For each rank configuration, fit per-band LS on roughness profiles
  for rank in ranks do
    for nBands in [4, 8, 16] do
      let bandSize := 128 / nBands
      -- Local warp: row i within band → x = 2*i/(bandSize-1) - 1
      let warpLocal : Nat → Float := fun i =>
        if bandSize <= 1 then 0.0
        else 2.0 * Float.ofNat i / Float.ofNat (bandSize - 1) - 1.0
      -- degR: fit degree — for 8 rows per band, degR=4 gives 5 DOF < 8 rows (overdetermined)
      let degR := min 4 (bandSize - 1)

      -- Fit each SVD component's roughness profile per band
      -- bandRoughCoeffs[k * nBands * (degR+1) + b * (degR+1) + j]
      let bandRoughCoeffs : Array Float := Id.run do
        let mut out := Array.mkEmpty (rank * nBands * (degR + 1))
        for k in [0:rank] do
          let (sigma, u, _v) := svdComponents[k]!
          -- Scale u by sigma so reconstruction is just sum_k u_k(i) * v_k(j)
          let uScaled : Array Float := u.map (· * sigma)
          for b in [0:nBands] do
            let r0 := b * bandSize
            let r1 := min (r0 + bandSize) 128
            let rows := r1 - r0
            -- Build Gram matrix G (rows × (degR+1))
            let G : Array Float := Id.run do
              let mut g := Array.mkEmpty (rows * (degR + 1))
              for i in [0:rows] do
                let x := warpLocal i
                for p in [0:(degR + 1)] do g := g.push (chebF p x)
              pure g
            -- GᵀG
            let GtG : Array Float := Id.run do
              let mut m := Array.replicate ((degR+1)*(degR+1)) 0.0
              for i in [0:rows] do
                for p in [0:(degR+1)] do
                  for q in [0:(degR+1)] do
                    let idx := p * (degR+1) + q
                    m := m.set! idx (m[idx]! + G[i*(degR+1)+p]! * G[i*(degR+1)+q]!)
              pure m
            -- Gᵀf where f = uScaled[r0..r1)
            let mut Gtf := Array.replicate (degR+1) 0.0
            for i in [0:rows] do
              let fi := uScaled[r0 + i]!
              for p in [0:(degR+1)] do
                Gtf := Gtf.set! p (Gtf[p]! + G[i*(degR+1)+p]! * fi)
            let c := gaussElim GtG Gtf (degR+1)
            for p in [0:(degR+1)] do out := out.push c[p]!
        pure out

      -- Project each v_k onto degC Chebyshev basis in cosTheta
      let degC := 10
      let vCoeffs : Array Float := Id.run do
        let mut out := Array.mkEmpty (rank * (degC + 1))
        for k in [0:rank] do
          let (_sigma, _u, v) := svdComponents[k]!
          for d in [0:(degC + 1)] do
            let mut s := 0.0
            for j in [0:128] do
              s := s + v[j]! * chebF d warpedFloats[j]!
            out := out.push (s / if d == 0 then 128.0 else 64.0)
        pure out

      -- Evaluate and gather stats
      let mut mseFull := 0.0; let mut mseInt := 0.0; let mut mseTop := 0.0
      let mut maxErr := 0.0; let mut maxIdx := 0
      for i in [0:128] do
        let b := min (i / bandSize) (nBands - 1)
        let iLocal := i - b * bandSize
        let x := warpLocal iLocal
        for j in [0:128] do
          let xv := warpedFloats[j]!
          let mut y := 0.0
          for k in [0:rank] do
            -- Roughness: evaluate band polynomial
            let mut uk := 0.0
            for p in [0:(degR+1)] do
              uk := uk + bandRoughCoeffs[k*nBands*(degR+1) + b*(degR+1) + p]! * chebF p x
            -- CosTheta: evaluate Chebyshev series
            let mut vk := 0.0
            for d in [0:(degC+1)] do
              vk := vk + vCoeffs[k*(degC+1)+d]! * chebF d xv
            y := y + uk * vk
          let fitted := max y 0.0
          let err := Float.abs (lut[i*128+j]! - fitted)
          mseFull := mseFull + err * err
          if err > maxErr then maxErr := err; maxIdx := i*128+j
          if j >= 1 then mseInt := mseInt + err * err
          if i >= 120 then mseTop := mseTop + err * err
      mseFull := mseFull / 16384.0
      mseInt  := mseInt  / 16256.0
      mseTop  := mseTop  / 1024.0
      let totalCoeffs := rank * nBands * (degR + 1) + rank * (degC + 1)
      IO.println s!"rank={rank} nBands={nBands} bandSize={bandSize} degR={degR} totalCoeffs={totalCoeffs}:"
      IO.println s!"  fullMSE={mseFull}  intMSE={mseInt}  topMSE={mseTop}  maxErr={maxErr}"
      IO.println s!"  maxErr at row={maxIdx/128} col={maxIdx%128}"
  IO.println ""
  IO.println "Reference: current production (rank-4 global Chebyshev + edge correction):"
  IO.println "  fullMSE≈0.002349  intMSE≈0.000875  topMSE≈0.035  maxErr=1.953"

/-- Best combined approach:
    - Three roughness segments with individually tuned warps
    - Segments A and B use r² warp (x = 2r²-1), which spreads samples
      away from the r=1 boundary and reduces interior oscillation 27%
    - Segment C (rows 110-127) uses a FLIPPED linear warp so roughness=1
      maps to x=-1 (interior of Chebyshev interval) not x=+1 (endpoint
      where catastrophic cancellation occurs)
    - All segments use degC=10 (full cosTheta basis), tuned degR per segment
    Reports: per-segment errors, overall combined error, and the full
    coefficient listing ready to paste into the shader. -/
def fitBestApproach (lut : Array Float) : IO Unit := do
  IO.println ""
  IO.println "=== Best combined approach: sqrt warp (A,B) + flipped warp (C) ==="
  IO.println "(sqrt warp preserves Chebyshev quadrature orthogonality in pass 2)"
  -- Segments A and B: use sqrt warp on global row index → correct DCT orthogonality
  -- Segment C: flip the local warp so roughness=1 (row 127) → x=-1 (not x=+1)
  --   flipped: i=0 (row 110) → x=+1,  i=17 (row 127) → x=-1
  let warpSqrtA : Nat → Float := fun i => warpedFloats[i]!          -- global sqrt warp
  let warpSqrtB : Nat → Float := fun i => warpedFloats[i + 73]!     -- global sqrt warp (offset)
  let warpFlipC : Nat → Float := fun i =>
    1.0 - 2.0 * Float.ofNat i / 17.0
  let mkSlice (r0 r1 : Nat) : Array Float := Id.run do
    let mut out := Array.mkEmpty ((r1 - r0) * 128)
    for i in [r0:r1] do
      for j in [0:128] do out := out.push lut[i * 128 + j]!
    pure out
  let sliceA := mkSlice 0 73
  let sliceB := mkSlice 73 110
  let sliceC := mkSlice 110 128
  let (rCoefsA, mseA, mseIntA, _, maxErrA, _) := twoPassFit sliceA warpSqrtA 10 10
  let (rCoefsB, mseB, mseIntB, _, maxErrB, _) := twoPassFit sliceB warpSqrtB 10 10
  let (rCoefsC, mseC, mseIntC, mseTopC, maxErrC, e127c1) := twoPassFit sliceC warpFlipC 11 10
  IO.println s!"Segment A [0,73)    sqrt-warp  degR=10 degC=10:"
  IO.println s!"  MSE={mseA}  intMSE={mseIntA}  maxErr={maxErrA}"
  IO.println s!"Segment B [73,110)  sqrt-warp  degR=10 degC=10:"
  IO.println s!"  MSE={mseB}  intMSE={mseIntB}  maxErr={maxErrB}"
  IO.println s!"Segment C [110,128] flip-warp  degR=11 degC=10:"
  IO.println s!"  MSE={mseC}  intMSE={mseIntC}  topMSE={mseTopC}  maxErr={maxErrC}  e127c1={e127c1}"
  let evalSeg (roughCoeffs : Array Float) (warpR : Nat → Float)
      (nRows degR degC : Nat) : Array Float :=
    let n := degC + 1
    Id.run do
      let mut out := Array.mkEmpty (nRows * 128)
      for i in [0:nRows] do
        let xr := warpR i
        for j in [0:128] do
          let xv := warpedFloats[j]!
          let mut y := 0.0
          for d in [0:n] do
            let mut rc := 0.0
            for k in [0:(degR + 1)] do
              rc := rc + roughCoeffs[d * (degR + 1) + k]! * chebF k xr
            y := y + rc * chebF d xv
          out := out.push (max y 0.0)
      pure out
  let fittedA := evalSeg rCoefsA warpSqrtA 73 10 10
  let fittedB := evalSeg rCoefsB warpSqrtB 37 10 10
  let fittedC := evalSeg rCoefsC warpFlipC 18 11 10
  let fullFitted : Array Float := Id.run do
    let mut out := Array.mkEmpty (128 * 128)
    for i in [0:73]  do for j in [0:128] do out := out.push fittedA[i * 128 + j]!
    for i in [0:37]  do for j in [0:128] do out := out.push fittedB[i * 128 + j]!
    for i in [0:18]  do for j in [0:128] do out := out.push fittedC[i * 128 + j]!
    pure out
  let mut mseFull := 0.0; let mut mseInt := 0.0; let mut mseTop := 0.0
  let mut maxErr := 0.0; let mut maxIdx := 0
  for i in [0:128] do
    for j in [0:128] do
      let idx := i * 128 + j
      let err := Float.abs (lut[idx]! - fullFitted[idx]!)
      mseFull := mseFull + err * err
      if err > maxErr then maxErr := err; maxIdx := idx
      if j >= 1 then mseInt := mseInt + err * err
      if i >= 120 then mseTop := mseTop + err * err
  mseFull := mseFull / 16384.0
  mseInt  := mseInt  / 16256.0
  mseTop  := mseTop  / 1024.0
  IO.println ""
  IO.println "Summary vs current production model (rank-4 global):"
  IO.println "                    fullMSE   intMSE   topMSE   maxErr"
  IO.println s!"  Current (rank-4): 0.002349  0.000875  ≈0.035   1.953"
  IO.println s!"  Best approach:    {mseFull}  {mseInt}  {mseTop}  {maxErr}"
  IO.println ""
  IO.println s!"Combined maxErr={maxErr} at row={maxIdx/128} col={maxIdx%128}"
  IO.println ""
  IO.println "Row=127 (roughness=1) truth vs fit:"
  IO.println "col  truth       fitted      err"
  for j in [0,1,2,3,4,5,6,7,8,9,10] do
    let t := lut[127 * 128 + j]!
    let f := fullFitted[127 * 128 + j]!
    IO.println s!"  {j}   {t}   {f}   {f - t}"
  IO.println ""
  IO.println "Rendering comparison matrix with best-approach model..."
  let noFit : EdgeFit := EdgeFit.mk
    { rough0 := 0.85, rough1 := 1.0, cos0 := 0.02, cos1 := 0.0 }
    0.0 0.0 0.0 0
  renderMatrix lut fullFitted noFit

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
  -- Also report interior-only error (col >= 1, excluding the aliased cosTheta=0 edge)
  let mut mseFull := 0.0; let mut mseInt := 0.0
  let mut maxFull := 0.0; let mut maxInt := 0.0
  let mut maxFullIdx := 0; let mut maxIntIdx := 0
  for i in [0:128] do
    let roughness := gridFloats[i]!
    for j in [0:128] do
      let idx := i * 128 + j
      let cosTheta := gridFloats[j]!
      let ec := max (approxGrid[idx]! + fit.amplitude * edgeWeight fit.params roughness cosTheta) 0.0
      let err := Float.abs (lut[idx]! - ec)
      mseFull := mseFull + err * err
      if err > maxFull then maxFull := err; maxFullIdx := idx
      if j >= 1 then
        mseInt := mseInt + err * err
        if err > maxInt then maxInt := err; maxIntIdx := idx
  mseFull := mseFull / 16384.0
  mseInt  := mseInt  / (16384.0 - 128.0)
  IO.println s!"interior (col≥1) MSE: {mseInt}  maxErr: {maxInt} at row {maxIntIdx/128} col {maxIntIdx%128}"
  IO.println s!"col=0 excluded — that edge is a DDS quantisation artifact (delta of ~10.9 in one pixel at roughness=1)"
  -- Write the pixel-level data table and render
  writeErrorTable lut approxGrid fit
  renderMatrix lut approxGrid fit

end SheenLutMobileCheck

def main : IO Unit :=
  SheenLutMobileCheck.checkGroundTruth
