# Overnight fetch + channel-ablation sweep for f1-telemetry-intelligence.
#
# What this does, in order:
#   1. Disables sleep/monitor timeout so an overnight run can't die the way
#      victorious-quail-905 did (22h "training" that was really the laptop
#      asleep).
#   2. Re-ingests the two races we already validate against (2024 Bahrain,
#      2024 Monza) using the new get_telemetry()-based ingest.py, so they
#      carry DRS/X/Y/Z position too, not just the original 5 channels.
#   3. Fetches every completed 2026 race via fetch_season.py.
#   4. Combines all of that into one telemetry_all.parquet.
#   5. Trains 4 channel configs back-to-back (baseline, +Acceleration,
#      +LateralAcceleration, +both) and validates each against real
#      race-control incidents, logging every run ID and result.
#
# Run from the repo root, with the venv activated:
#   .\scripts\overnight_run.ps1
#
# In the morning, check logs\overnight_<timestamp>.log for the full record,
# or just scroll up in this terminal if it's still open.

$ErrorActionPreference = "Continue"

Write-Host "Disabling sleep/monitor timeout for this run..."
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-ac 0

$logDir = "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmm"
$log = Join-Path $logDir "overnight_$stamp.log"

function Log-Line($msg) {
    $line = "$(Get-Date -Format 'HH:mm:ss')  $msg"
    Write-Host $line
    Add-Content -Path $log -Value $line
}

function Run-Logged($cmdLine) {
    Log-Line ">> $cmdLine"
    $output = Invoke-Expression $cmdLine 2>&1
    $output | ForEach-Object { Add-Content -Path $log -Value $_; Write-Host $_ }
    return $output
}

Log-Line "=== Overnight run started ==="

Log-Line "Step 1/5: re-ingesting 2024 Bahrain with full channel set (DRS/position)..."
Run-Logged "python -m f1telemetry.data.ingest 2024 Bahrain --session R --mode full --out data/raw/bahrain2024_full.parquet"

Log-Line "Step 1/5: re-ingesting 2024 Monza with full channel set (DRS/position)..."
Run-Logged "python -m f1telemetry.data.ingest 2024 Monza --session R --mode full --out data/raw/monza2024_full.parquet"

Log-Line "Step 2/5: fetching every completed 2026 race (this is the long part)..."
Run-Logged "python scripts/fetch_season.py 2026"

Log-Line "Step 3/5: combining everything into telemetry_all.parquet..."
Run-Logged "python scripts/combine_telemetry.py bahrain2024_full monza2024_full --dir data/raw/season_2026 --out data/raw/telemetry_all.parquet"

$telemetry = "data/raw/telemetry_all.parquet"

$configs = @(
    @{ Name = "baseline_5ch";  Channels = "Speed Throttle Brake nGear RPM" },
    @{ Name = "plus_accel";    Channels = "Speed Throttle Brake nGear RPM Acceleration" },
    @{ Name = "plus_lataccel"; Channels = "Speed Throttle Brake nGear RPM LateralAcceleration" },
    @{ Name = "plus_both";     Channels = "Speed Throttle Brake nGear RPM Acceleration LateralAcceleration" }
)

foreach ($cfg in $configs) {
    Log-Line "Step 4/5: training config '$($cfg.Name)' (channels: $($cfg.Channels))..."
    $trainCmd = "python -m f1telemetry.models.train_lstm `"$telemetry`" " +
        "--window-size 20 --stride 10 --hidden-size 64 --num-layers 2 --epochs 50 " +
        "--channels $($cfg.Channels) " +
        "--register-as f1-anomaly-lstm-$($cfg.Name)"
    $trainOutput = Run-Logged $trainCmd

    $runIdLine = $trainOutput | Select-String "MLflow run ID: (\S+)"
    if ($runIdLine) {
        $runId = $runIdLine.Matches[0].Groups[1].Value
        Log-Line "  -> run ID: $runId. Validating against real incidents..."
        Run-Logged "python scripts/validate_anomalies.py `"$telemetry`" --run-id $runId"
    } else {
        Log-Line "  -> could not parse a run ID from training output, skipping validation for $($cfg.Name)."
    }
}

Log-Line "=== Overnight run finished. Full record: $log ==="
Log-Line "Look for '(z = ...)' and 'Verdict:' lines for each config's result."
