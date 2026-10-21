$ErrorActionPreference='Stop'
$taskRoot=Split-Path $PSScriptRoot -Parent
. (Join-Path $PSScriptRoot 'docker_path.ps1')
$taskDocker=Get-GuanyeDocker
$taskPython=Join-Path $taskRoot '.venv-platform/Scripts/python.exe'
& $taskPython (Join-Path $PSScriptRoot 'compose_run.py') --docker $taskDocker compose --project-directory $taskRoot -f (Join-Path $taskRoot 'compose.yaml') --profile models --profile research logs --tail 160 models mlflow
if($LASTEXITCODE -ne 0){throw 'Docker log collection failed.'}
Write-Host 'Research logs saved in .runtime/compose-last-start.log. No rebuild was performed.'
