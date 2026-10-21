$ErrorActionPreference='Stop'
$taskRoot=Split-Path $PSScriptRoot -Parent
. (Join-Path $PSScriptRoot 'docker_path.ps1')
$taskDocker=Get-GuanyeDocker
$taskGitPath=Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd'
if((-not(Get-Command git -ErrorAction SilentlyContinue)) -and (Test-Path -LiteralPath (Join-Path $taskGitPath 'git.exe'))){$env:PATH="$taskGitPath;C:\Windows\System32;$env:PATH"}
$taskPython=Join-Path $taskRoot '.venv-platform/Scripts/python.exe'
if(-not(Test-Path -LiteralPath $taskPython)){throw 'Run scripts/bootstrap_python.ps1 first.'}
if(-not(Test-Path -LiteralPath (Join-Path $taskRoot '.env'))){throw 'Run scripts/init_config.py first.'}
$taskCore=Invoke-RestMethod -Uri 'http://127.0.0.1:18081/actuator/health' -TimeoutSec 5
if($taskCore.status -ne 'UP'){throw 'Start the core services with start_platform_services.bat first.'}
& $taskDocker image inspect guanye-worker --format '{{.Id}}'
if($LASTEXITCODE -ne 0){throw 'The existing core worker image is missing. Run start_platform_services.bat first.'}
$taskArgs=@('compose','--project-directory',$taskRoot,'-f',(Join-Path $taskRoot 'compose.yaml'),'-f',(Join-Path $taskRoot 'compose.public.yaml'),'--profile','research','--profile','models')
& $taskDocker @taskArgs config --quiet
if($LASTEXITCODE -ne 0){throw 'Compose config validation failed.'}
# Build only the new model image; reuse the already verified worker image for MLflow.
& $taskPython (Join-Path $PSScriptRoot 'compose_run.py') --docker $taskDocker @taskArgs --progress plain build models
if($LASTEXITCODE -ne 0){throw ('Model build failed; inspect '+(Join-Path $taskRoot '.runtime/compose-last-start.log'))}
& $taskPython (Join-Path $PSScriptRoot 'compose_run.py') --docker $taskDocker @taskArgs --progress plain up -d --no-deps --no-build mlflow models
if($LASTEXITCODE -ne 0){throw ('Research startup failed; inspect '+(Join-Path $taskRoot '.runtime/compose-last-start.log'))}
$taskReady=$false
for($taskAttempt=0;$taskAttempt -lt 60;$taskAttempt++){
    try{
        $taskModels=Invoke-RestMethod -Uri 'http://127.0.0.1:18082/health' -TimeoutSec 2
        $taskTracking=Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:15000/health' -TimeoutSec 2
        if($taskModels.ready -and $taskTracking.StatusCode -eq 200){$taskReady=$true;break}
    }catch{}
    Start-Sleep -Seconds 1
}
if(-not $taskReady){
    & $taskPython (Join-Path $PSScriptRoot 'compose_run.py') --docker $taskDocker @taskArgs logs --tail 80 mlflow models
    throw 'Research services have not become healthy. Preserve the output for diagnosis.'
}
Write-Host 'Research services ready: DINOv2/SAM 2.1 on CPU, MLflow on port 15000.'
