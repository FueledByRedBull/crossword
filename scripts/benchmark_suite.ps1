param(
  [string]$Seeds = "Thermodynamics,Jazz,Ancient Rome",
  [string]$OutputDir = "outputs/benchmarks_suite",
  [switch]$Offline
)

if ($Offline) {
  $env:CROSSWORD_OFFLINE = "1"
}

python scripts/bench.py `
  --seeds "$Seeds" `
  --lang en `
  --cache-dir data/cache/wiki `
  --output-dir $OutputDir `
  --max-links 1000 `
  --max-backlinks 1000 `
  --expansion one_hop_plus_bounded_two_hop `
  --max-two-hop-parents 12 `
  --max-two-hop-links 60 `
  --max-candidates 1500 `
  --min-df 1 `
  --gate-max 250 `
  --grid-size 15 `
  --min-slot-len 3 `
  --max-steps 80000 `
  --max-restarts 5
