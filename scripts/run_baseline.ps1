param(
    [string]$Python = "python",
    [string[]]$Models = @(
        "DSAINet",
        "EEGNet",
        "ShallowConvNet",
        "DeepConvNet",
        "ADFCNN",
        "CTNet",
        "LMDANet",
        "MSCFormer",
        "TMSANet",
        "Deformer"
    ),
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

$Models = @($Models | ForEach-Object { $_ -split "," } | Where-Object { $_ -ne "" })
$Datasets = @($Datasets | ForEach-Object { $_ -split "," } | Where-Object { $_ -ne "" })
$Seeds = @($Seeds | ForEach-Object { $_ -split "," } | Where-Object { $_ -ne "" })

$ConfigByModel = @{
    "DSAINet" = "config/DSAINet.yaml"
    "EEGNet" = "config/EEGNet.yaml"
    "ShallowConvNet" = "config/ShallowConvNet.yaml"
    "DeepConvNet" = "config/DeepConvNet.yaml"
    "ADFCNN" = "config/ADFCNN.yaml"
    "CTNet" = "config/CTNet.yaml"
    "LMDANet" = "config/LMDANet.yaml"
    "MSCFormer" = "config/MSCFormer.yaml"
    "TMSANet" = "config/TMSANet.yaml"
    "Deformer" = "config/Deformer.yaml"
}
$LosoDatasets = @("BCIC-IV-2a", "BCIC-IV-2b", "Zhou2016")

foreach ($model in $Models) {
    if (-not $ConfigByModel.ContainsKey($model)) {
        throw "Unknown baseline model: $model"
    }
    foreach ($dataset in $Datasets) {
        $trainer = if ($LosoDatasets -contains $dataset) { "train_loso.py" } else { "train_kfold.py" }
        foreach ($seed in $Seeds) {
            $args = @(
                $trainer,
                "--config", $ConfigByModel[$model],
                "--dataset", $dataset,
                "--seed", "$seed",
                "--device", "$Device"
            )
            Write-Host "[$model][$dataset][seed=$seed] $Python $($args -join ' ')"
            if ($DryRun) {
                continue
            }
            & $Python @args
            if ($LASTEXITCODE -ne 0) {
                $message = "Failed: model=$model dataset=$dataset seed=$seed exit=$LASTEXITCODE"
                if ($StopOnError) {
                    throw $message
                }
                Write-Warning $message
            }
        }
    }
}
