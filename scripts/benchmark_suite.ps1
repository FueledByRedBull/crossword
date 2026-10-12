param(
  [string]$Seeds = "Thermodynamics,Quantum mechanics,Jazz,Ancient Rome",
  [string]$Lang = "en",
  [string]$Fixture = "data/benchmarks/gate_a.json",
  [string]$OutputDir = "outputs/benchmarks_suite",
  [switch]$Offline
)

if ($Offline) {
  $env:CROSSWORD_OFFLINE = "1"
}

py -3 scripts/bench.py `
  --seeds "$Seeds" `
  --lang $Lang `
  --gate-a-fixture $Fixture `
  --cache-dir data/cache/wiki `
  --output-dir $OutputDir
