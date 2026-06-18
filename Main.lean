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
  -- Shared scale: 99th-percentile of LUT values (avoids one bright corner swamping the palette)
  let sortedLut := lut.toList.mergeSort (· ≤ ·) |>.toArray
  let p99       := (lut.size * 99) / 100
  let valMax    := max (sortedLut[p99]!) 0.001
  -- Four cell pixel arrays
  let tlPixels := lut.map     greyPixel                          -- top-left:  GT grey
  let trPixels := approxEC.map greyPixel                         -- top-right: approx grey
  let blPixels := lut.map     (fun v => falseColor (v / valMax)) -- bot-left:  GT false
  let brPixels := approxEC.map (fun v => falseColor (v / valMax))-- bot-right: approx false
  IO.FS.createDirAll "rendered"
  -- Write temp PPMs
  writePpm "rendered/_tl.ppm" tlPixels 128 128
  writePpm "rendered/_tr.ppm" trPixels 128 128
  writePpm "rendered/_bl.ppm" blPixels 128 128
  writePpm "rendered/_br.ppm" brPixels 128 128
  -- Compose with ImageMagick: +append = horizontal, -append = vertical
  -- Scale each cell up 4x, add a solid label bar below, then compose
  -- Cell: 128x128 -> 512x512, label bar: 512x32
  let labelCell : String → String → String → IO Unit := fun src lbl dst =>
    runConvert #[src,
      "-scale", "512x512",
      "-gravity", "South",
      "-background", "black",
      "-splice", "0x32",           -- add 32px strip at bottom
      "-fill", "white",
      "-font", "DejaVu-Sans-Bold", "-pointsize", "20",
      "-gravity", "South", "-annotate", "+0+6", lbl,
      dst]
  labelCell "rendered/_tl.ppm" "Ground truth"         "rendered/_tl_l.ppm"
  labelCell "rendered/_tr.ppm" "Best approx (EC)"     "rendered/_tr_l.ppm"
  labelCell "rendered/_bl.ppm" "Ground truth (false)"  "rendered/_bl_l.ppm"
  labelCell "rendered/_br.ppm" "Best approx (false)"   "rendered/_br_l.ppm"
  runConvert #["rendered/_tl_l.ppm", "rendered/_tr_l.ppm", "+append", "rendered/_top.ppm"]
  runConvert #["rendered/_bl_l.ppm", "rendered/_br_l.ppm", "+append", "rendered/_bot.ppm"]
  runConvert #["rendered/_top.ppm", "rendered/_bot.ppm", "-append", "rendered/matrix_2x2.png"]
  -- Clean up all temp files
  for f in ["rendered/_tl.ppm",   "rendered/_tr.ppm",
            "rendered/_bl.ppm",   "rendered/_br.ppm",
            "rendered/_tl_l.ppm", "rendered/_tr_l.ppm",
            "rendered/_bl_l.ppm", "rendered/_br_l.ppm",
            "rendered/_top.ppm",  "rendered/_bot.ppm"] do
    rmTemp f
  IO.println "wrote rendered/matrix_2x2.png  (GT grey | approx grey // GT false | approx false)"

/-- Dot product of two flat vectors. -/
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
  renderMatrix lut approxGrid fit
  reportNumericalRank lut
  fitPiecewise lut
  profileRoughnessRows lut

end SheenLutMobileCheck

def main : IO Unit :=
  SheenLutMobileCheck.checkGroundTruth
