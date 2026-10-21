$ErrorActionPreference='Stop';$taskRoot=Split-Path $PSScriptRoot -Parent
$taskUv=Get-Command uv -ErrorAction SilentlyContinue
if(-not $taskUv -and (Test-Path -LiteralPath (Join-Path $taskRoot '.tooling/uv/uv-0.13.3.data/scripts/uv.exe'))){$taskUv=Join-Path $taskRoot '.tooling/uv/uv-0.13.3.data/scripts/uv.exe'}
if(-not $taskUv){throw 'Install uv from the official Astral release or PyPI, then reopen the terminal. This script does not change your existing GPU environment.'}
Push-Location -LiteralPath $taskRoot
try { & $taskUv venv --python 3.11 .venv-platform; if($LASTEXITCODE -ne 0){throw 'uv environment creation failed'}; & $taskUv pip install --python .venv-platform -e '.[experiments]' opencv-python-headless kafka-python-ng; if($LASTEXITCODE -ne 0){throw 'uv dependency installation failed'}; & '.venv-platform/Scripts/python.exe' scripts/init_config.py; & '.venv-platform/Scripts/python.exe' scripts/catalog.py; & '.venv-platform/Scripts/python.exe' scripts/make_gateway.py }
finally{Pop-Location}
