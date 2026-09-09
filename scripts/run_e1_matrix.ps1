param(
    [string]$Networks = 'siouxfalls ema anaheim',
    [string]$Seeds = $(if ($env:SEEDS) { $env:SEEDS } else { '42 43 44 45 46' }),
    [string]$PythonExe = 'python',
    [string]$Device = 'cuda'
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
foreach ($network in ($Networks -split '[,\s]+' | Where-Object { $_ })) {
    & $PythonExe -B scripts/e1_information_set.py --network $network --mode persistence --device $Device
    if ($LASTEXITCODE -ne 0) { throw "Persistence failed: $network" }
    foreach ($mode in @('od_only', 'old_state', 'hybrid')) {
        foreach ($seed in ($Seeds -split '[,\s]+' | Where-Object { $_ })) {
            & $PythonExe -B scripts/e1_information_set.py --network $network --mode $mode --seed $seed --device $Device
            if ($LASTEXITCODE -ne 0) { throw "E1 failed: $network $mode $seed" }
        }
    }
}
