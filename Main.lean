import SheenLutProof

open SheenLutProof

def main : IO Unit := do
  IO.println s!"rank components: {components.length}"
  IO.println s!"degree per factor: {degree}"
  IO.println s!"witness DAG nodes: {witnessDag.length}"
  IO.println s!"component coefficient lengths ok: {components.all componentDegreeOk}"
  IO.println s!"DAG acyclic: {witnessDag.all (fun n => n.deps.all (fun d => d < n.idx))}"
  IO.println s!"sample lutApproxAt(0,0): {lutApproxAt 0 0}"
