<#
.SYNOPSIS
    Disk cleanup helper for the AI Swarm project.

.DESCRIPTION
    Reports sizes of common dev-machine disk hogs and (with -Clean) removes
    the safe ones. Targets:
      - the broken partial DeepSeek download (definite waste)
      - HuggingFace cache for other models you may not use anymore (interactive)
      - pip wheel cache
      - npm cache
      - conda package cache
      - Windows TEMP folders
      - __pycache__ directories inside this project
      - Recycle Bin

    Run once without -Clean to see what's there; review; then run with -Clean.

.PARAMETER Clean
    Actually delete. Without this switch, the script only reports.

.PARAMETER SkipHFInteractive
    Skip the interactive Hugging Face cache scanner. Useful if you don't have
    huggingface_hub installed or just want the automated steps.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\cleanup_disk.ps1
    powershell -ExecutionPolicy Bypass -File .\cleanup_disk.ps1 -Clean
#>

[CmdletBinding()]
param(
    [switch]$Clean,
    [switch]$SkipHFInteractive
)

$ErrorActionPreference = "Continue"

function Get-DirSize {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return 0 }
    try {
        $sum = (Get-ChildItem -LiteralPath $Path -Recurse -Force -ErrorAction SilentlyContinue |
                Measure-Object -Property Length -Sum).Sum
        if ($null -eq $sum) { return 0 }
        return [int64]$sum
    } catch { return 0 }
}

function Format-Size {
    param([int64]$Bytes)
    if ($Bytes -ge 1GB) { return ("{0:N2} GB" -f ($Bytes / 1GB)) }
    if ($Bytes -ge 1MB) { return ("{0:N2} MB" -f ($Bytes / 1MB)) }
    if ($Bytes -ge 1KB) { return ("{0:N2} KB" -f ($Bytes / 1KB)) }
    return "$Bytes B"
}

function Get-DriveFree {
    param([string]$DriveLetter = "C")
    $drive = Get-PSDrive -Name $DriveLetter -ErrorAction SilentlyContinue
    if ($null -eq $drive) { return 0 }
    return [int64]$drive.Free
}

function Remove-IfExists {
    param([string]$Path, [string]$Label)
    if (-not (Test-Path -LiteralPath $Path)) {
        Write-Host "  [skip]  $Label  (not present)"
        return
    }
    $size = Get-DirSize -Path $Path
    Write-Host "  [---]   $Label  ($(Format-Size $size))  -- $Path"
    if ($Clean) {
        try {
            Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction Stop
            Write-Host "  [done]  removed ($(Format-Size $size) reclaimed)"
        } catch {
            Write-Host "  [warn]  could not remove fully: $($_.Exception.Message)"
        }
    }
}

# ---------------------------------------------------------------------------

$startFree = Get-DriveFree -DriveLetter "C"
Write-Host ""
Write-Host "================================================================"
Write-Host "  Disk cleanup helper"
Write-Host "  Mode: $(if ($Clean) { 'CLEAN (will delete)' } else { 'DRY RUN (report only)' })"
Write-Host "  C: free at start: $(Format-Size $startFree)"
Write-Host "================================================================"

# 1. Partial DeepSeek download (the immediate culprit)
Write-Host ""
Write-Host "[1] Partial DeepSeek-R1-Distill-Qwen-7B download (interrupted, broken)"
$hfRoot = Join-Path $env:USERPROFILE ".cache\huggingface\hub"
$partial = Join-Path $hfRoot "models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B"
Remove-IfExists -Path $partial -Label "DeepSeek partial download"

# 2. Other HF cache models (interactive review)
Write-Host ""
Write-Host "[2] Other Hugging Face cached models"
if (Test-Path -LiteralPath $hfRoot) {
    $models = Get-ChildItem -LiteralPath $hfRoot -Directory -ErrorAction SilentlyContinue |
              Where-Object { $_.Name -like "models--*" }
    if ($models) {
        foreach ($m in $models) {
            $size = Get-DirSize -Path $m.FullName
            Write-Host "  [info]  $($m.Name)  ($(Format-Size $size))"
        }
        if (-not $SkipHFInteractive) {
            Write-Host ""
            Write-Host "  To pick which models to drop interactively, run:"
            Write-Host "    pip install huggingface_hub  # if not already"
            Write-Host "    huggingface-cli delete-cache"
        }
    } else {
        Write-Host "  [skip]  no models cached"
    }
} else {
    Write-Host "  [skip]  no HF cache directory"
}

# 3. pip wheel cache
Write-Host ""
Write-Host "[3] pip wheel cache"
$pipCache = Join-Path $env:LOCALAPPDATA "pip\Cache"
Remove-IfExists -Path $pipCache -Label "pip wheel cache"

# 4. npm cache
Write-Host ""
Write-Host "[4] npm cache"
$npmCache = Join-Path $env:APPDATA "npm-cache"
Remove-IfExists -Path $npmCache -Label "npm cache (APPDATA)"
$npmCache2 = Join-Path $env:USERPROFILE ".npm"
Remove-IfExists -Path $npmCache2 -Label "npm cache (USERPROFILE)"

# 5. conda package cache (only the cache, NOT envs)
Write-Host ""
Write-Host "[5] conda package cache (envs not touched)"
foreach ($condaRoot in @(
    "$env:USERPROFILE\miniconda3\pkgs",
    "$env:USERPROFILE\anaconda3\pkgs",
    "$env:LOCALAPPDATA\Continuum\miniconda3\pkgs"
)) {
    Remove-IfExists -Path $condaRoot -Label "conda pkgs: $condaRoot"
}

# 6. Windows TEMP
Write-Host ""
Write-Host "[6] Windows TEMP folders"
foreach ($t in @($env:TEMP, "$env:LOCALAPPDATA\Temp")) {
    if (Test-Path -LiteralPath $t) {
        $size = Get-DirSize -Path $t
        Write-Host "  [---]   $t  ($(Format-Size $size))"
        if ($Clean) {
            Get-ChildItem -LiteralPath $t -Force -ErrorAction SilentlyContinue | ForEach-Object {
                try { Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue } catch {}
            }
            $after = Get-DirSize -Path $t
            Write-Host "  [done]  $(Format-Size ($size - $after)) reclaimed"
        }
    }
}

# 7. __pycache__ in this project
Write-Host ""
Write-Host "[7] __pycache__ inside this project"
$projectRoot = $PSScriptRoot
if (-not $projectRoot) { $projectRoot = "." }
$pycaches = Get-ChildItem -LiteralPath $projectRoot -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue
$totalPyc = 0
foreach ($pc in $pycaches) {
    $size = Get-DirSize -Path $pc.FullName
    $totalPyc += $size
}
Write-Host "  [info]  $($pycaches.Count) __pycache__ dirs totaling $(Format-Size $totalPyc)"
if ($Clean) {
    foreach ($pc in $pycaches) {
        try { Remove-Item -LiteralPath $pc.FullName -Recurse -Force -ErrorAction Stop } catch {}
    }
    Write-Host "  [done]  removed"
}

# 8. Recycle Bin
Write-Host ""
Write-Host "[8] Recycle Bin"
if ($Clean) {
    try {
        Clear-RecycleBin -Force -ErrorAction Stop
        Write-Host "  [done]  emptied"
    } catch {
        Write-Host "  [warn]  could not empty: $($_.Exception.Message)"
    }
} else {
    Write-Host "  [info]  will be emptied with -Clean"
}

# ---------------------------------------------------------------------------

$endFree = Get-DriveFree -DriveLetter "C"
$delta = $endFree - $startFree
Write-Host ""
Write-Host "================================================================"
Write-Host "  C: free at end:   $(Format-Size $endFree)"
if ($Clean) {
    Write-Host "  Space reclaimed:  $(Format-Size $delta)"
} else {
    Write-Host "  (dry run — re-run with -Clean to actually delete)"
}
Write-Host "================================================================"

# Optional aggressive steps the user can run manually if they need more space
Write-Host ""
Write-Host "If you still need more space, these require admin and take longer:"
Write-Host "  cleanmgr.exe /sageset:99      # configure deep cleanup"
Write-Host "  cleanmgr.exe /sagerun:99      # run it"
Write-Host "  DISM.exe /Online /Cleanup-Image /StartComponentCleanup /ResetBase"
Write-Host ""
