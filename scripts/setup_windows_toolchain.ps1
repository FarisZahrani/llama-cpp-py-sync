param(
    [switch]$Quiet,
    [string]$VsWherePath = "",
    [string]$Command = ""
)

$ErrorActionPreference = "Stop"

function Resolve-VsWhere {
    if ($VsWherePath -and (Test-Path -LiteralPath $VsWherePath)) {
        return (Resolve-Path -LiteralPath $VsWherePath).Path
    }

    $pf86 = [Environment]::GetEnvironmentVariable("ProgramFiles(x86)")
    $pf = [Environment]::GetEnvironmentVariable("ProgramFiles")

    $candidates = @()
    if ($pf86) {
        $candidates += (Join-Path $pf86 "Microsoft Visual Studio\Installer\vswhere.exe")
    }
    if ($pf) {
        $candidates += (Join-Path $pf "Microsoft Visual Studio\Installer\vswhere.exe")
    }

    foreach ($p in $candidates) {
        if ($p -and (Test-Path -LiteralPath $p)) {
            return (Resolve-Path -LiteralPath $p).Path
        }
    }

    throw "vswhere.exe not found. Install Visual Studio Build Tools 2022 (C++ workload)."
}

function Get-VsInstallPath([string]$vswhere) {
    $install = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    $install = ($install | Select-Object -First 1)
    if (-not $install) {
        throw "Visual Studio Build Tools with MSVC (x64/x86) not found. Install the C++ workload."
    }
    return $install.Trim()
}

function Import-VcVars([string]$vcvarsBat) {
    if (-not (Test-Path -LiteralPath $vcvarsBat)) {
        throw "vcvars64.bat not found: $vcvarsBat"
    }

    $out = & cmd /c "`"$vcvarsBat`" && set"
    foreach ($line in $out) {
        $idx = $line.IndexOf('=')
        if ($idx -le 0) { continue }
        $name = $line.Substring(0, $idx)
        $value = $line.Substring($idx + 1)
        Set-Item -Path "Env:$name" -Value $value
    }
}

$vswhere = Resolve-VsWhere
$installPath = Get-VsInstallPath $vswhere

$vcvars64 = Join-Path $installPath "VC\Auxiliary\Build\vcvars64.bat"
Import-VcVars $vcvars64

# Visual Studio often bundles CMake+Ninja here; make them visible if present.
$cmakeBin = Join-Path $installPath "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"
$ninjaBin = Join-Path $installPath "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
if (Test-Path -LiteralPath $cmakeBin) { $env:Path = "$cmakeBin;" + $env:Path }
if (Test-Path -LiteralPath $ninjaBin) { $env:Path = "$ninjaBin;" + $env:Path }

if (-not $Quiet) {
    Write-Host "MSVC toolchain loaded from: $installPath"
    $cl = (Get-Command cl.exe -ErrorAction SilentlyContinue)
    $cmake = (Get-Command cmake.exe -ErrorAction SilentlyContinue)
    $ninja = (Get-Command ninja.exe -ErrorAction SilentlyContinue)
    if ($cl -and $cl.Path) { Write-Host "cl.exe:    " $cl.Path } else { Write-Host "cl.exe:    " "<not found>" }
    if ($cmake -and $cmake.Path) { Write-Host "cmake.exe: " $cmake.Path } else { Write-Host "cmake.exe: " "<not found>" }
    if ($ninja -and $ninja.Path) { Write-Host "ninja.exe: " $ninja.Path } else { Write-Host "ninja.exe: " "<not found>" }
}

if ($Command) {
    & powershell -NoProfile -Command $Command
}
