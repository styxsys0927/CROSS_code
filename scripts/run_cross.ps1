param(
    [string]$Python = "python",
    [string[]]$Datasets = @(
        "BCIC-IV-2a",
        "BCIC-IV-2b",
        "Zhou2016",
        "OpenBMI",
        "PhysioNet-MI",
        "Mumtaz2017",
        "ADFTD",
        "Rockhill2021",
        "EEGMat",
        "Shin2018"
    ),
    [string[]]$Seeds = @("0"),
    [int]$Device = 0,
    [switch]$DryRun,
    [switch]$StopOnError
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

$Datasets = @($Datasets | ForEach-Object { $_ -split "," } | Where-Object { $_ -ne "" })
$Seeds = @($Seeds | ForEach-Object { $_ -split "," } | Where-Object { $_ -ne "" })
$LosoDatasets = @("BCIC-IV-2a", "BCIC-IV-2b", "Zhou2016")

foreach ($dataset in $Datasets) {
    $trainer = if ($LosoDatasets -contains $dataset) { "train_loso.py" } else { "train_kfold.py" }
    foreach ($seed in $Seeds) {
        $args = @(
            $trainer,
            "--config", "config/cross.yaml",
            "--dataset", $dataset,
            "--seed", "$seed",
            "--device", "$Device"
        )
        Write-Host "[CROSS][$dataset][seed=$seed] $Python $($args -join ' ')"
        if ($DryRun) {
            continue
        }
        & $Python @args
        if ($LASTEXITCODE -ne 0) {
            $message = "Failed: model=CROSS dataset=$dataset seed=$seed exit=$LASTEXITCODE"
            if ($StopOnError) {
                throw $message
            }
            Write-Warning $message
        }
    }
}
