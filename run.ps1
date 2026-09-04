<#
.SYNOPSIS
    Windows equivalent of the Makefile. `make` is a Unix tool and is not on a
    stock Windows box, so every target has a PowerShell twin here.

.DESCRIPTION
    Same target names as the Makefile, same order, same artifacts. Use whichever
    matches your shell:

        make test          ->   .\run.ps1 test
        make clean-memory  ->   .\run.ps1 clean-memory

.EXAMPLE
    .\run.ps1 all
    .\run.ps1 test
    .\run.ps1 run
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('help', 'setup', 'data', 'graph', 'candidates', 'features', 'train',
                 'all', 'small', 'test', 'lint', 'run', 'demo', 'docker',
                 'clean', 'clean-memory', 'reset-demo')]
    [string]$Target = 'help'
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
Set-Location $root

# PYTHONPATH so `python -m src.foo` resolves from anywhere, matching the
# `export PYTHONPATH := $(CURDIR)` line in the Makefile.
$env:PYTHONPATH = $root

# Prefer the venv interpreter; fall back to whatever python is on PATH so this
# still works in CI, which installs into the runner's system Python.
$venvPy = Join-Path $root '.venv\Scripts\python.exe'
$py = if (Test-Path $venvPy) { $venvPy } else { 'python' }

function Invoke-Py {
    <#
      .SYNOPSIS
        Run the chosen interpreter and stop the script if it fails.
      .DESCRIPTION
        PowerShell does not abort on a non-zero exit code from a native exe, so
        a failing stage would otherwise let the next one run against stale or
        missing artifacts.
    #>
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$PyArgs)
    Write-Host ">> python $($PyArgs -join ' ')" -ForegroundColor DarkGray
    & $py @PyArgs
    if ($LASTEXITCODE -ne 0) { throw "python $($PyArgs -join ' ') failed (exit $LASTEXITCODE)" }
}

function Remove-IfPresent {
    <#
      .SYNOPSIS
        Delete paths that may not exist, without erroring — the `rm -f` the
        Makefile relies on.
    #>
    param([string[]]$Paths)
    foreach ($p in $Paths) {
        Get-ChildItem -Path $p -ErrorAction SilentlyContinue |
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    }
}

switch ($Target) {

    'help' {
        Write-Host ''
        Write-Host 'Ringfence - Abuse-Ring Sentinel' -ForegroundColor Cyan
        Write-Host '  .\run.ps1 setup         create .venv and install requirements'
        Write-Host '  .\run.ps1 all           full pipeline: data -> graph -> candidates -> features -> model'
        Write-Host '  .\run.ps1 test          run the test suite (114 tests)'
        Write-Host '  .\run.ps1 run           start the API on :8000'
        Write-Host '  .\run.ps1 demo          start the dashboard on :8501'
        Write-Host '  .\run.ps1 reset-demo    re-arm the failure demo, KEEPING the alert queue'
        Write-Host '  .\run.ps1 clean-memory  reset case memory AND the alert queue'
        Write-Host '  .\run.ps1 clean         remove all generated artifacts'
        Write-Host ''
        Write-Host ('interpreter: {0}' -f $py) -ForegroundColor DarkGray
        Write-Host ''
    }

    'setup' {
        & python -m venv .venv
        Invoke-Py -m pip install --upgrade pip
        Invoke-Py -m pip install -r requirements.txt
        Write-Host ''
        Write-Host 'Setup complete. Copy .env.example to .env if you want the LLM summary layer.' -ForegroundColor Green
    }

    'data'       { Invoke-Py -m src.data_generator }
    'graph'      { Invoke-Py -m src.graph_builder }
    'candidates' { Invoke-Py -m src.candidate_generator }
    'features'   { Invoke-Py -m src.feature_engine }
    'train'      { Invoke-Py -m src.scorer }

    'all' {
        Invoke-Py -m src.data_generator
        Invoke-Py -m src.graph_builder
        Invoke-Py -m src.candidate_generator
        Invoke-Py -m src.feature_engine
        Invoke-Py -m src.scorer
        Write-Host ''
        Write-Host 'Pipeline complete. Artifacts in data/processed/.' -ForegroundColor Green
        Write-Host 'Now: .\run.ps1 run   (then, in a second terminal)   .\run.ps1 demo'
    }

    'small' {
        Invoke-Py -m src.data_generator --small
        Invoke-Py -m src.graph_builder
        Invoke-Py -m src.candidate_generator
        Invoke-Py -m src.feature_engine
        Invoke-Py -m src.scorer
    }

    'test' { Invoke-Py -m pytest }
    'lint' { Invoke-Py -m compileall -q src frontend tests }

    'run'  { Invoke-Py -m uvicorn src.api:app --host 0.0.0.0 --port 8000 }
    'demo' { Invoke-Py -m streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0 }

    'docker' {
        docker compose up --build
        if ($LASTEXITCODE -ne 0) { throw "docker compose failed (exit $LASTEXITCODE)" }
    }

    'clean' {
        Remove-IfPresent @('data\raw\*.csv', 'data\processed\*', 'data\ground_truth\*.csv')
        Write-Host "Artifacts removed. Run '.\run.ps1 all' to rebuild." -ForegroundColor Green
    }

    'clean-memory' {
        Remove-IfPresent @('data\processed\case_memory.sqlite', 'data\processed\alerts.sqlite')
        Write-Host 'Case memory cleared - the failure-recovery demo is re-armed.' -ForegroundColor Green
        Write-Host 'The alert queue is now empty: run  python demo\seed_alerts.py --limit 150' -ForegroundColor Yellow
    }

    'reset-demo' {
        # Between takes: re-arm the failure demo and drop analyst dispositions
        # without destroying the seeded queue, which costs minutes to rebuild.
        Invoke-Py demo\reset_demo.py
    }
}
