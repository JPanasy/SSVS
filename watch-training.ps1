param(
    [int]$IntervalSeconds = 5,
    [int]$TailLines = 8,
    [switch]$Once
)

$ErrorActionPreference = "SilentlyContinue"

$logCandidates = @(
    ".\checkpoints\check\loss_log.txt",
    ".\checkpoints\ssv_bs_probe\loss_log.txt"
)

function Get-LogSummary {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        return "[missing] $Path"
    }

    $item = Get-Item $Path
    $stamp = $item.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
    $size = $item.Length
    $tail = Get-Content $Path -Tail $TailLines

    $body = if ($tail) { ($tail -join "`n") } else { "(empty)" }
    return "[log] $Path`nLastWrite: $stamp  Size: $size bytes`n$body"
}

function Get-GpuSummary {
    $nvsmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $nvsmi) {
        return "nvidia-smi not found on PATH"
    }

    $query = "name,utilization.gpu,memory.used,memory.total,temperature.gpu"
    $lines = & nvidia-smi --query-gpu=$query --format=csv,noheader,nounits 2>$null

    if (-not $lines) {
        return "GPU query returned no data"
    }

    $formatted = @()
    foreach ($line in $lines) {
        $parts = $line -split ",\s*"
        if ($parts.Count -ge 5) {
            $formatted += "{0}: util {1}% | mem {2}/{3} MiB | temp {4}C" -f $parts[0], $parts[1], $parts[2], $parts[3], $parts[4]
        } else {
            $formatted += $line
        }
    }

    return ($formatted -join "`n")
}

do {
    Clear-Host
    Write-Host ("SSVS training monitor  " + (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")) -ForegroundColor Cyan
    Write-Host ("-" * 70)

    $existing = $logCandidates | Where-Object { Test-Path $_ }
    if (-not $existing) {
        Write-Host "No known loss logs found yet." -ForegroundColor Yellow
        Write-Host "Expected one of:" -ForegroundColor Yellow
        $logCandidates | ForEach-Object { Write-Host "  $_" -ForegroundColor Yellow }
    } else {
        foreach ($path in $existing) {
            Write-Host (Get-LogSummary -Path $path)
            Write-Host ""
        }
    }

    Write-Host "GPU"
    Write-Host (Get-GpuSummary)

    if (-not $Once) {
        Start-Sleep -Seconds $IntervalSeconds
    }
} while (-not $Once)
