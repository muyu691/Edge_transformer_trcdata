param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('SiouxFalls', 'EMA', 'Anaheim')][string]$Network,
    [ValidateRange(1, 1000000)][int]$TargetPairs = 7000,
    [ValidateRange(1, 1000)][int]$BatchSize = 250,
    [ValidateRange(0, 1000000)][int]$Seed = 42,
    [Parameter(Mandatory=$true)][string]$OutputDir,
    [string]$PythonExe = 'python'
)

# Each batch owns a short-lived Python process: no dataset-sized worker copies
# and no growing in-memory list across batches. Outputs stay as separate shards.
$ErrorActionPreference = 'Stop'
$projectDir = Split-Path $PSScriptRoot -Parent
$solverDir = Join-Path $projectDir 'create_sioux_data'
$solverFile = Join-Path $solverDir 'solve_network_pairs.py'
$outputRoot = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDir)
& $PythonExe -c "import sys; from pathlib import Path; assert Path(sys.prefix).name.lower() == 'graphgps', 'Activate graphgps or pass -PythonExe for that environment'"
if ($LASTEXITCODE -ne 0) { throw 'Python environment check failed.' }

$checkShard = @'
import pickle, sys
sys.path.insert(0, sys.argv[2])
from solve_network_pairs import validate_sue_certificates
with open(sys.argv[1], 'rb') as handle:
    pairs = pickle.load(handle)['pairs']
validate_sue_certificates(pairs)
assert all(p['network_name'].lower() == sys.argv[3].lower() for p in pairs), 'Wrong network in shard'
print(len(pairs))
'@

$accepted = 0
$batch = 0
$emptyBatches = 0
while ($accepted -lt $TargetPairs) {
    $batchSeed = $Seed + $batch
    $batchDir = Join-Path $outputRoot ('batch_{0:D4}_seed_{1}' -f ($batch + 1), $batchSeed)
    $datasetFile = Join-Path $batchDir 'network_pairs_dataset.pkl'
    if (-not (Test-Path -LiteralPath $datasetFile)) {
        # Do not silently overwrite a partially completed batch. Its saved base
        # scenarios/flows may be needed for an explicit recovery.
        if (Test-Path -LiteralPath $batchDir) {
            throw "Incomplete batch: $batchDir. Recover it, or move this batch directory aside before rerunning."
        }
        $requested = [Math]::Min($BatchSize, $TargetPairs - $accepted)
        Write-Host "[$Network] batch $($batch + 1): $requested candidates, accepted $accepted/$TargetPairs"
        & $PythonExe -B $solverFile --network_name $Network --num_samples $requested `
            --seed $batchSeed --output_dir $batchDir --num_workers 1 `
            --sue_loading_protocol reasonable_links --convergence_threshold 1e-5 `
            --retry_max_iter 5000 --checkpoint_interval 0
        if ($LASTEXITCODE -ne 0) { throw "Generation failed: $batchDir" }
    }
    # Also validate existing completed shards when this command is restarted.
    $countText = & $PythonExe -B -c $checkShard $datasetFile $solverDir $Network
    if ($LASTEXITCODE -ne 0) { throw "Shard verification failed: $datasetFile" }
    $count = [int]$countText
    if ($count -eq 0) {
        $emptyBatches++
        if ($emptyBatches -ge 3) { throw 'Three consecutive empty batches; inspect failures before continuing.' }
        Write-Warning "No accepted pairs in $batchDir; trying the next seed."
    } else { $emptyBatches = 0 }
    if ($accepted + $count -gt $TargetPairs) {
        throw 'Existing shards exceed this target. Use the original target or a new output directory.'
    }
    $accepted += $count
    $batch++
    Write-Host "[$Network] verified $accepted/$TargetPairs pairs in $batch shards."
}
Write-Host "Complete: $outputRoot ($accepted valid pairs). Keep shards separate to limit RAM use."
