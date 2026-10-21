$ErrorActionPreference='Stop'
$taskRoot=Split-Path $PSScriptRoot -Parent
. (Join-Path $PSScriptRoot 'docker_path.ps1')
$taskDocker=Get-GuanyeDocker
$taskPython=Join-Path $taskRoot '.venv-platform/Scripts/python.exe'
& $taskPython (Join-Path $PSScriptRoot 'compose_run.py') --docker $taskDocker compose --project-directory $taskRoot -f (Join-Path $taskRoot 'compose.yaml') --progress plain up -d --no-deps --build business
if($LASTEXITCODE -ne 0){throw ('Business update failed; inspect '+(Join-Path $taskRoot '.runtime/compose-last-start.log'))}
$taskReady=$false
for($taskAttempt=0;$taskAttempt -lt 30;$taskAttempt++){
    try {
        $taskHealth=Invoke-RestMethod -Uri 'http://127.0.0.1:18081/actuator/health' -TimeoutSec 2
        if($taskHealth.status -eq 'UP'){$taskReady=$true;break}
    }catch{}
    Start-Sleep -Seconds 1
}
if(-not $taskReady){throw 'Business container started but health is not UP yet. Preserve the output for diagnosis.'}
Write-Host 'Business updated and health is UP.'
