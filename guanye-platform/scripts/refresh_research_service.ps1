$ErrorActionPreference='Stop'
$taskRoot=Split-Path $PSScriptRoot -Parent
. (Join-Path $PSScriptRoot 'docker_path.ps1')
$taskDocker=Get-GuanyeDocker
$taskPython=Join-Path $taskRoot '.venv-platform/Scripts/python.exe'
& $taskPython (Join-Path $PSScriptRoot 'compose_run.py') --docker $taskDocker compose --project-directory $taskRoot -f (Join-Path $taskRoot 'compose.yaml') -f (Join-Path $taskRoot 'compose.public.yaml') --profile models up -d --no-deps --no-build models
if($LASTEXITCODE -ne 0){throw 'Model container refresh failed; inspect .runtime/compose-last-start.log.'}
Write-Host 'Model runtime configuration refreshed. No image build or package download was performed.'
