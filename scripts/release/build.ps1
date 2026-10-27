<#
.SYNOPSIS
    Build OrcaGym distribution packages on Windows.

.DESCRIPTION
    Windows counterpart of scripts/release/build.sh: runs the production build
    in dist/ and then the TestPyPI build in dist-test/.

.PARAMETER Python
    Python interpreter used for every build step. Defaults to "python" on PATH;
    pass the orca environment interpreter when the environment is not activated.

.EXAMPLE
    .\scripts\release\build.ps1

.EXAMPLE
    .\scripts\release\build.ps1 -Python C:\Users\me\miniforge3\envs\orca\python.exe
#>
[CmdletBinding()]
param(
    [string] $Python = 'python'
)

$ErrorActionPreference = 'Stop'

$ScriptDir = $PSScriptRoot
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir '..\..')).Path
$TestOutDir = Join-Path $ProjectRoot 'dist-test'

function Resolve-PythonInterpreter {
    $command = Get-Command $Python -ErrorAction SilentlyContinue
    if ($command) {
        $interpreter = $command.Source
    } elseif (Test-Path -LiteralPath $Python) {
        $interpreter = (Resolve-Path -LiteralPath $Python).Path
    } else {
        throw "Python interpreter not found: $Python"
    }

    # Console scripts spawned by the build live next to the interpreter, so make
    # them reachable even without activating the environment.
    $interpreterDir = Split-Path -Parent $interpreter
    $env:PATH = "$interpreterDir;$(Join-Path $interpreterDir 'Scripts');$env:PATH"
    return $interpreter
}

$script:PythonExe = Resolve-PythonInterpreter

function Invoke-Python {
    param([Parameter(Mandatory = $true)][string[]] $Arguments)

    & $script:PythonExe @Arguments | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "Python step failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

function Test-BuildModuleAvailable {
    & $script:PythonExe -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('build') else 1)"
    return ($LASTEXITCODE -eq 0)
}

function Copy-BuildTree {
    param([Parameter(Mandatory = $true)][string] $Destination)

    $robocopy = Get-Command robocopy -ErrorAction SilentlyContinue
    if (-not $robocopy) {
        throw 'robocopy is required to stage the TestPyPI build tree but was not found.'
    }

    # Mirror the rsync excludes used by build.sh: VCS data, previous build
    # outputs and packaging metadata never enter the TestPyPI build tree.
    & $robocopy.Source @(
        $ProjectRoot,
        $Destination,
        '/E',
        '/XD', '.git', 'dist', 'dist-test', 'build', '*.egg-info',
        '/XF', '.gitignore',
        '/NJH', '/NJS', '/NFL', '/NDL', '/NP'
    ) | Out-Null
    if ($LASTEXITCODE -gt 7) {
        throw "robocopy failed with exit code $LASTEXITCODE"
    }
}

Push-Location -LiteralPath $ProjectRoot
try {
    Write-Host 'Building distribution packages...'

    if (-not (Test-BuildModuleAvailable)) {
        Write-Host "'build' not found. Installing..."
        Invoke-Python -Arguments @('-m', 'pip', 'install', 'build')
    }

    # Production build (official name), writing sdist and wheel into dist/.
    Invoke-Python -Arguments @('-m', 'build')

    Write-Host ''
    Write-Host 'Building TestPyPI package (orca-gym)...'

    # Keep the same package name (orca-gym) for the TestPyPI build so
    # installation is exercised exactly like the production package.
    $tempDir = Join-Path ([System.IO.Path]::GetTempPath()) ('orcagym-testbuild-' + [Guid]::NewGuid().ToString('N'))
    try {
        New-Item -ItemType Directory -Path $tempDir | Out-Null
        Copy-BuildTree -Destination $tempDir

        New-Item -ItemType Directory -Path $TestOutDir -Force | Out-Null
        Push-Location -LiteralPath $tempDir
        try {
            Invoke-Python -Arguments @('-m', 'build', '--outdir', $TestOutDir)
        } finally {
            Pop-Location
        }
    } finally {
        if (Test-Path -LiteralPath $tempDir) {
            Remove-Item -LiteralPath $tempDir -Recurse -Force
        }
    }

    Write-Host ''
    Write-Host 'Build completed!'
    Write-Host ''
    Write-Host 'Generated files (dist/):'
    Get-ChildItem -LiteralPath (Join-Path $ProjectRoot 'dist') -File |
        Format-Table Name, Length -AutoSize | Out-Host
    Write-Host 'Generated files for TestPyPI (dist-test/):'
    Get-ChildItem -LiteralPath $TestOutDir -File |
        Format-Table Name, Length -AutoSize | Out-Host
} catch {
    Write-Error $_
    exit 1
} finally {
    Pop-Location
}
