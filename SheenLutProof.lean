/-!
# Sheen LUT separable Chebyshev proof

This module is the Lean4 proof artifact for the generated approximation. The
coefficient generator fits the square root of the LUT with a rank-8, degree-20
separable Chebyshev model, rounds coefficients to 7 decimal places, then squares
`max(raw, 0)` to recover the LUT domain.

Lean treats the rounded coefficients as the exact shader contract and proves that
its Flowref-style plausible witness DAG computes the same expression as the direct
separable Chebyshev specification.
-/

namespace SheenLutProof

/-- Number of separable SVD witnesses in the generated model. -/
def rankCount : Nat := 8

/-- Chebyshev degree per univariate factor. -/
def degree : Nat := 20

/-- The common fixed-point scale for rounded coefficients. -/
def coefficientScale : Nat := 10000000

/-- One low-rank component: `rough(r) * cosTheta(c)`. -/
structure Component where
  rank : Nat
  roughCoeffs : List Rat
  cosThetaCoeffs : List Rat
  deriving Repr, DecidableEq

/-- A DAG node witnessing one recovered low-rank component. -/
structure WitnessNode where
  idx : Nat
  component : Component
  deps : List Nat
  deriving Repr, DecidableEq

/-- The query/outcome vocabulary mirrors Flowref's plausible witness DAG. -/
inductive Outcome
  | found (witnessIdx : Nat)
  | provablyNone
  | budgetHit
  deriving Repr, DecidableEq

structure TraceEntry where
  query : String
  level : Nat
  outcome : Outcome
  deriving Repr, DecidableEq

/-- Chebyshev polynomial `T_n(x)` over exact rationals. -/
def cheb : Nat → Rat → Rat
  | 0, _ => 1
  | 1, x => x
  | n + 2, x => 2 * x * cheb (n + 1) x - cheb n x

/-- Pair each coefficient with the Chebyshev order it multiplies. -/
def coeffTerms (coeffs : List Rat) : List (Nat × Rat) :=
  List.zip (List.range coeffs.length) coeffs

/-- Exact rational Chebyshev series evaluation. -/
def evalChebSeries (coeffs : List Rat) (x : Rat) : Rat :=
  (coeffTerms coeffs).foldl (fun acc p => acc + p.2 * cheb p.1 x) 0

/-- Evaluation of one separable component. -/
def Component.eval (component : Component) (roughness cosTheta : Rat) : Rat :=
  evalChebSeries component.roughCoeffs roughness *
    evalChebSeries component.cosThetaCoeffs cosTheta

/-- Rounded coefficients produced by the SVD/Chebyshev generator. -/
def components : List Component := [
  { rank := 0, roughCoeffs := [((-181775117711203 : Rat) / 10000000), ((307783163534144 : Rat) / 10000000), ((-159212894166904 : Rat) / 10000000), ((-34031823582890 : Rat) / 10000000), ((214406355093362 : Rat) / 10000000), ((-336776517401345 : Rat) / 10000000), ((382067888610012 : Rat) / 10000000), ((-358114083261648 : Rat) / 10000000), ((290036830073587 : Rat) / 10000000), ((-206853727678888 : Rat) / 10000000), ((130949364332262 : Rat) / 10000000), ((-73718930707106 : Rat) / 10000000), ((36815809224282 : Rat) / 10000000), ((-16209308974140 : Rat) / 10000000), ((6226943581987 : Rat) / 10000000), ((-2054884085714 : Rat) / 10000000), ((569127941155 : Rat) / 10000000), ((-127657863516 : Rat) / 10000000), ((21864662985 : Rat) / 10000000), ((-2558977513 : Rat) / 10000000), ((154843155 : Rat) / 10000000)], cosThetaCoeffs := [((-394457721574414 : Rat) / 10000000), ((666039188463984 : Rat) / 10000000), ((-339708935908984 : Rat) / 10000000), ((-81859198834864 : Rat) / 10000000), ((470563246614057 : Rat) / 10000000), ((-728016206670909 : Rat) / 10000000), ((815472453098430 : Rat) / 10000000), ((-754023987322221 : Rat) / 10000000), ((601419925818952 : Rat) / 10000000), ((-421548601802718 : Rat) / 10000000), ((261653193746421 : Rat) / 10000000), ((-144047116144220 : Rat) / 10000000), ((70145744970540 : Rat) / 10000000), ((-30016612066210 : Rat) / 10000000), ((11166200349145 : Rat) / 10000000), ((-3553180450568 : Rat) / 10000000), ((944265597245 : Rat) / 10000000), ((-202031402202 : Rat) / 10000000), ((32766979699 : Rat) / 10000000), ((-3597672756 : Rat) / 10000000), ((201622714 : Rat) / 10000000)] },
  { rank := 1, roughCoeffs := [((850015161906642 : Rat) / 10000000), ((-1439243838261514 : Rat) / 10000000), ((744481551534023 : Rat) / 10000000), ((159179198943531 : Rat) / 10000000), ((-1002632646298656 : Rat) / 10000000), ((1574819607942723 : Rat) / 10000000), ((-1786556292276375 : Rat) / 10000000), ((1674493994670412 : Rat) / 10000000), ((-1356123566456513 : Rat) / 10000000), ((967143057916884 : Rat) / 10000000), ((-612221911921068 : Rat) / 10000000), ((344634665640070 : Rat) / 10000000), ((-172101444575240 : Rat) / 10000000), ((75766982400129 : Rat) / 10000000), ((-29103820739322 : Rat) / 10000000), ((9603183162456 : Rat) / 10000000), ((-2659392712925 : Rat) / 10000000), ((596423676269 : Rat) / 10000000), ((-102134430792 : Rat) / 10000000), ((11950905759 : Rat) / 10000000), ((-722942443 : Rat) / 10000000)], cosThetaCoeffs := [((1920425326839083 : Rat) / 10000000), ((-3242635567472030 : Rat) / 10000000), ((1653913342274734 : Rat) / 10000000), ((398483171180915 : Rat) / 10000000), ((-2290906854335806 : Rat) / 10000000), ((3544366991335459 : Rat) / 10000000), ((-3970220446375038 : Rat) / 10000000), ((3671130889595484 : Rat) / 10000000), ((-2928225723544964 : Rat) / 10000000), ((2052531619080560 : Rat) / 10000000), ((-1274054206358006 : Rat) / 10000000), ((701440886025291 : Rat) / 10000000), ((-341600302538311 : Rat) / 10000000), ((146189271060326 : Rat) / 10000000), ((-54388024931786 : Rat) / 10000000), ((17308788116330 : Rat) / 10000000), ((-4600471747403 : Rat) / 10000000), ((984445890062 : Rat) / 10000000), ((-159689581849 : Rat) / 10000000), ((17535704376 : Rat) / 10000000), ((-982827984 : Rat) / 10000000)] },
  { rank := 2, roughCoeffs := [((-1910082315939846 : Rat) / 10000000), ((3234088237948076 : Rat) / 10000000), ((-1672751312948357 : Rat) / 10000000), ((-357952966548841 : Rat) / 10000000), ((2253208966627292 : Rat) / 10000000), ((-3538723049845707 : Rat) / 10000000), ((4014167418068101 : Rat) / 10000000), ((-3762030709157942 : Rat) / 10000000), ((3046436607361582 : Rat) / 10000000), ((-2172352692270066 : Rat) / 10000000), ((1374948428221757 : Rat) / 10000000), ((-773862932958575 : Rat) / 10000000), ((386371294539122 : Rat) / 10000000), ((-170059469141515 : Rat) / 10000000), ((65306028483845 : Rat) / 10000000), ((-21541619195267 : Rat) / 10000000), ((5963159623301 : Rat) / 10000000), ((-1336721721660 : Rat) / 10000000), ((228768590465 : Rat) / 10000000), ((-26747458469 : Rat) / 10000000), ((1616229996 : Rat) / 10000000)], cosThetaCoeffs := [((-3485382441531436 : Rat) / 10000000), ((5885052008126135 : Rat) / 10000000), ((-3001658557158688 : Rat) / 10000000), ((-723236197037628 : Rat) / 10000000), ((4157761686372514 : Rat) / 10000000), ((-6432617567676457 : Rat) / 10000000), ((7205473834932331 : Rat) / 10000000), ((-6662680091727443 : Rat) / 10000000), ((5314446845448416 : Rat) / 10000000), ((-3725223120422862 : Rat) / 10000000), ((2312411981692424 : Rat) / 10000000), ((-1273183176435362 : Rat) / 10000000), ((620083447685032 : Rat) / 10000000), ((-265394091761967 : Rat) / 10000000), ((98750029144226 : Rat) / 10000000), ((-31432297848095 : Rat) / 10000000), ((8356189253784 : Rat) / 10000000), ((-1788625727338 : Rat) / 10000000), ((290239701528 : Rat) / 10000000), ((-31885590747 : Rat) / 10000000), ((1788088861 : Rat) / 10000000)] },
  { rank := 3, roughCoeffs := [((2214520837562842 : Rat) / 10000000), ((-3749205706911354 : Rat) / 10000000), ((1938274238090080 : Rat) / 10000000), ((416530413313164 : Rat) / 10000000), ((-2613381954436477 : Rat) / 10000000), ((4102278338557197 : Rat) / 10000000), ((-4651425524091676 : Rat) / 10000000), ((4357219432328643 : Rat) / 10000000), ((-3526518163991044 : Rat) / 10000000), ((2513121905903864 : Rat) / 10000000), ((-1589476437592198 : Rat) / 10000000), ((893847698787412 : Rat) / 10000000), ((-445834638269569 : Rat) / 10000000), ((196003822239759 : Rat) / 10000000), ((-75165857344475 : Rat) / 10000000), ((24753446086019 : Rat) / 10000000), ((-6838772436255 : Rat) / 10000000), ((1529305419883 : Rat) / 10000000), ((-260935556645 : Rat) / 10000000), ((30388341353 : Rat) / 10000000), ((-1826175827 : Rat) / 10000000)], cosThetaCoeffs := [((3438575615413708 : Rat) / 10000000), ((-5806102218857857 : Rat) / 10000000), ((2961616939261664 : Rat) / 10000000), ((713123032115927 : Rat) / 10000000), ((-4101607143980116 : Rat) / 10000000), ((6346275487341393 : Rat) / 10000000), ((-7109320316239834 : Rat) / 10000000), ((6574403656638486 : Rat) / 10000000), ((-5244682789889998 : Rat) / 10000000), ((3676907731294900 : Rat) / 10000000), ((-2282885754530457 : Rat) / 10000000), ((1257249943050739 : Rat) / 10000000), ((-612519970303531 : Rat) / 10000000), ((262260666964099 : Rat) / 10000000), ((-97631202830708 : Rat) / 10000000), ((31094284999285 : Rat) / 10000000), ((-8272100407483 : Rat) / 10000000), ((1772097041971 : Rat) / 10000000), ((-287840082708 : Rat) / 10000000), ((31658582628 : Rat) / 10000000), ((-1777778511 : Rat) / 10000000)] },
  { rank := 4, roughCoeffs := [((1300777110843721 : Rat) / 10000000), ((-2204000941530257 : Rat) / 10000000), ((1144058484003378 : Rat) / 10000000), ((236905846500743 : Rat) / 10000000), ((-1529764811279409 : Rat) / 10000000), ((2412015811953058 : Rat) / 10000000), ((-2745129674949514 : Rat) / 10000000), ((2581851958398184 : Rat) / 10000000), ((-2099196454283280 : Rat) / 10000000), ((1503860195135956 : Rat) / 10000000), ((-956948231161663 : Rat) / 10000000), ((541934203249255 : Rat) / 10000000), ((-272507708073285 : Rat) / 10000000), ((120933413384922 : Rat) / 10000000), ((-46885666999835 : Rat) / 10000000), ((15638532281371 : Rat) / 10000000), ((-4386114392438 : Rat) / 10000000), ((998674039659 : Rat) / 10000000), ((-174187584196 : Rat) / 10000000), ((20854679455 : Rat) / 10000000), ((-1300271069 : Rat) / 10000000)], cosThetaCoeffs := [((-1921895543816648 : Rat) / 10000000), ((3244991337214211 : Rat) / 10000000), ((-1654818408115066 : Rat) / 10000000), ((-399192156445301 : Rat) / 10000000), ((2292745738313924 : Rat) / 10000000), ((-3546611332493459 : Rat) / 10000000), ((3972331484047348 : Rat) / 10000000), ((-3672926122879722 : Rat) / 10000000), ((2929769710541048 : Rat) / 10000000), ((-2053924478509711 : Rat) / 10000000), ((1275304806660516 : Rat) / 10000000), ((-702479417645450 : Rat) / 10000000), ((342360561364645 : Rat) / 10000000), ((-146667701347182 : Rat) / 10000000), ((54642807307924 : Rat) / 10000000), ((-17421839017167 : Rat) / 10000000), ((4641382661557 : Rat) / 10000000), ((-996121039981 : Rat) / 10000000), ((162170933474 : Rat) / 10000000), ((-17887447438 : Rat) / 10000000), ((1007972205 : Rat) / 10000000)] },
  { rank := 5, roughCoeffs := [((-13650207743050352 : Rat) / 10000000), ((23114640312719828 : Rat) / 10000000), ((-11962150138209522 : Rat) / 10000000), ((-2546885277305743 : Rat) / 10000000), ((16094707690502934 : Rat) / 10000000), ((-25292619038285944 : Rat) / 10000000), ((28705546362049336 : Rat) / 10000000), ((-26917383891588896 : Rat) / 10000000), ((21811016253673560 : Rat) / 10000000), ((-15564275529865280 : Rat) / 10000000), ((9859353900990418 : Rat) / 10000000), ((-5554509616529946 : Rat) / 10000000), ((2776330781189610 : Rat) / 10000000), ((-1223567742274802 : Rat) / 10000000), ((470578294287178 : Rat) / 10000000), ((-155495628252927 : Rat) / 10000000), ((43133412365065 : Rat) / 10000000), ((-9692824173079 : Rat) / 10000000), ((1663833395350 : Rat) / 10000000), ((-195266118733 : Rat) / 10000000), ((11857372053 : Rat) / 10000000)], cosThetaCoeffs := [((174760933997919 : Rat) / 10000000), ((-294954494374193 : Rat) / 10000000), ((150159843367764 : Rat) / 10000000), ((36589076877420 : Rat) / 10000000), ((-208395719875983 : Rat) / 10000000), ((321894069195090 : Rat) / 10000000), ((-360345024964329 : Rat) / 10000000), ((333375881107278 : Rat) / 10000000), ((-266490455843018 : Rat) / 10000000), ((187625586984375 : Rat) / 10000000), ((-117329609221015 : Rat) / 10000000), ((65322225094052 : Rat) / 10000000), ((-32316983729810 : Rat) / 10000000), ((14126349591167 : Rat) / 10000000), ((-5401788741913 : Rat) / 10000000), ((1779402375774 : Rat) / 10000000), ((-493326272460 : Rat) / 10000000), ((111032461427 : Rat) / 10000000), ((-19110191796 : Rat) / 10000000), ((2247046983 : Rat) / 10000000), ((-136145345 : Rat) / 10000000)] },
  { rank := 6, roughCoeffs := [((31337609408959876 : Rat) / 10000000), ((-53057672320110992 : Rat) / 10000000), ((27437177744097840 : Rat) / 10000000), ((5882103130718120 : Rat) / 10000000), ((-36973551752066600 : Rat) / 10000000), ((58054985445206936 : Rat) / 10000000), ((-65842548758115432 : Rat) / 10000000), ((61694234973436416 : Rat) / 10000000), ((-49947303117224424 : Rat) / 10000000), ((35606613405288320 : Rat) / 10000000), ((-22529236460899140 : Rat) / 10000000), ((12675327013702326 : Rat) / 10000000), ((-6325654132095585 : Rat) / 10000000), ((2782728546369590 : Rat) / 10000000), ((-1067942962931118 : Rat) / 10000000), ((351998822466250 : Rat) / 10000000), ((-97349426727609 : Rat) / 10000000), ((21796763715319 : Rat) / 10000000), ((-3724768940135 : Rat) / 10000000), ((434632007359 : Rat) / 10000000), ((-26187809858 : Rat) / 10000000)], cosThetaCoeffs := [((1204306898799814 : Rat) / 10000000), ((-2034016366655001 : Rat) / 10000000), ((1038802290677113 : Rat) / 10000000), ((247850032615100 : Rat) / 10000000), ((-1435688972812688 : Rat) / 10000000), ((2224125682719920 : Rat) / 10000000), ((-2493752615524629 : Rat) / 10000000), ((2307743722867282 : Rat) / 10000000), ((-1841852925485553 : Rat) / 10000000), ((1291440887185886 : Rat) / 10000000), ((-801535519720597 : Rat) / 10000000), ((440986443503542 : Rat) / 10000000), ((-214448459497767 : Rat) / 10000000), ((91552876960685 : Rat) / 10000000), ((-33938221986659 : Rat) / 10000000), ((10745801851978 : Rat) / 10000000), ((-2836460768905 : Rat) / 10000000), ((601473064650 : Rat) / 10000000), ((-96425865661 : Rat) / 10000000), ((10430808191 : Rat) / 10000000), ((-573584744 : Rat) / 10000000)] },
  { rank := 7, roughCoeffs := [((-27848209612331868 : Rat) / 10000000), ((47138702150051872 : Rat) / 10000000), ((-24347463639557076 : Rat) / 10000000), ((-5275547637387570 : Rat) / 10000000), ((32889583649070772 : Rat) / 10000000), ((-51575550355690816 : Rat) / 10000000), ((58430153186547280 : Rat) / 10000000), ((-54684402816443440 : Rat) / 10000000), ((44212733750621280 : Rat) / 10000000), ((-31469618024736932 : Rat) / 10000000), ((19875846428998340 : Rat) / 10000000), ((-11159175874611116 : Rat) / 10000000), ((5555545320494958 : Rat) / 10000000), ((-2437072731339661 : Rat) / 10000000), ((932214155938093 : Rat) / 10000000), ((-306073709749990 : Rat) / 10000000), ((84259354113177 : Rat) / 10000000), ((-18761439702925 : Rat) / 10000000), ((3184272220690 : Rat) / 10000000), ((-368368477283 : Rat) / 10000000), ((21942013291 : Rat) / 10000000)], cosThetaCoeffs := [((-2110395984339846 : Rat) / 10000000), ((3564973074232323 : Rat) / 10000000), ((-1822320162766259 : Rat) / 10000000), ((-431530845535008 : Rat) / 10000000), ((2513833957213630 : Rat) / 10000000), ((-3898151271635238 : Rat) / 10000000), ((4374508072119670 : Rat) / 10000000), ((-4052245693882361 : Rat) / 10000000), ((3238084274247496 : Rat) / 10000000), ((-2273798756109164 : Rat) / 10000000), ((1413805096044672 : Rat) / 10000000), ((-779567543744732 : Rat) / 10000000), ((380116002570642 : Rat) / 10000000), ((-162804878232721 : Rat) / 10000000), ((60584818340635 : Rat) / 10000000), ((-19271531515954 : Rat) / 10000000), ((5114878582367 : Rat) / 10000000), ((-1091695728931 : Rat) / 10000000), ((176374681406 : Rat) / 10000000), ((-19255682382 : Rat) / 10000000), ((1070605033 : Rat) / 10000000)] }
]

/-- Each generated factor has exactly `degree + 1` Chebyshev coefficients. -/
def componentDegreeOk (component : Component) : Bool :=
  component.roughCoeffs.length == degree + 1 &&
    component.cosThetaCoeffs.length == degree + 1

/-- A compact witness DAG: each rank component depends only on earlier ranks. -/
def witnessDag : List WitnessNode :=
  (List.zip (List.range components.length) components).map (fun pair =>
    { idx := pair.1, component := pair.2, deps := List.range pair.1 })

/-- All component witnesses resolve at L0 for this generated model. -/
def witnessTrace : List TraceEntry :=
  witnessDag.map (fun n =>
    { query := s!"sqrt-space rank component {n.component.rank}", level := 0,
      outcome := .found n.idx })

/-- Direct separable Chebyshev specification, before clamp-and-square. -/
def rawSpec (roughness cosTheta : Rat) : Rat :=
  components.foldl (fun acc component => acc + component.eval roughness cosTheta) 0

/-- Witness-DAG evaluation, before clamp-and-square. -/
def rawFromWitnessDag (roughness cosTheta : Rat) : Rat :=
  witnessDag.foldl (fun acc n => acc + n.component.eval roughness cosTheta) 0

/-- Clamp negative square-root approximants before squaring back to LUT space. -/
def clampNonnegative (x : Rat) : Rat :=
  if x < 0 then 0 else x

/-- Final exact rational approximation corresponding to the shader expression. -/
def sheenApproxSpec (roughness cosTheta : Rat) : Rat :=
  let raw := rawSpec roughness cosTheta
  let v := clampNonnegative raw
  v * v

/-- Final exact rational approximation through the witness DAG. -/
def sheenApproxFromWitnessDag (roughness cosTheta : Rat) : Rat :=
  let raw := rawFromWitnessDag roughness cosTheta
  let v := clampNonnegative raw
  v * v

/-- Grid coordinate used by `np.linspace(0, 1, 128)`. -/
def gridCoord (idx : Nat) : Rat := (idx : Rat) / 127

/-- Lean version of `lut_approx[r,c]`. -/
def lutApproxAt (roughnessIdx cosThetaIdx : Nat) : Rat :=
  sheenApproxFromWitnessDag (gridCoord roughnessIdx) (gridCoord cosThetaIdx)

theorem components_length : components.length = rankCount := by native_decide

theorem components_degree_ok : components.all componentDegreeOk = true := by native_decide

theorem witnessDag_length : witnessDag.length = components.length := by native_decide

/-- Every witness node points only backward, so the list order is acyclic. -/
theorem witnessDag_acyclic :
    witnessDag.all (fun n => n.deps.all (fun d => d < n.idx)) = true := by
  native_decide

/-- The DAG payload list is exactly the generated component list. -/
theorem witnessDag_components : witnessDag.map (fun n => n.component) = components := by
  native_decide

theorem foldl_node_components (xs : List WitnessNode) (roughness cosTheta acc : Rat) :
    xs.foldl (fun acc n => acc + n.component.eval roughness cosTheta) acc =
      (xs.map (fun n => n.component)).foldl
        (fun acc component => acc + component.eval roughness cosTheta) acc := by
  induction xs generalizing acc with
  | nil => rfl
  | cons x xs ih => simp [List.foldl, ih]

/-- The core conversion proof: evaluating the plausible-style witness DAG is the
same exact rational raw approximation as the direct component specification. -/
theorem rawWitnessDag_correct (roughness cosTheta : Rat) :
    rawFromWitnessDag roughness cosTheta = rawSpec roughness cosTheta := by
  unfold rawFromWitnessDag rawSpec
  rw [foldl_node_components]
  rw [witnessDag_components]

/-- Clamp-and-square preserves the DAG/spec equality. -/
theorem sheenWitnessDag_correct (roughness cosTheta : Rat) :
    sheenApproxFromWitnessDag roughness cosTheta = sheenApproxSpec roughness cosTheta := by
  unfold sheenApproxFromWitnessDag sheenApproxSpec
  rw [rawWitnessDag_correct]

/-- The Lean grid evaluator is exactly the generated separable approximation. -/
theorem lutApproxAt_correct (roughnessIdx cosThetaIdx : Nat) :
    lutApproxAt roughnessIdx cosThetaIdx =
      sheenApproxSpec (gridCoord roughnessIdx) (gridCoord cosThetaIdx) := by
  unfold lutApproxAt
  exact sheenWitnessDag_correct (gridCoord roughnessIdx) (gridCoord cosThetaIdx)

end SheenLutProof
