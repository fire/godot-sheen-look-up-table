import Lake
open Lake DSL

package «godot-sheen-look-up-table» where
  version := v!"0.1.0"

lean_lib SheenLutProof where
  roots := #[`SheenLutProof]

lean_exe «sheen-lut-proof» where
  root := `Main
