$ErrorActionPreference='Stop';$taskRoot=Split-Path $PSScriptRoot -Parent
$taskPython=Join-Path $taskRoot '.venv-platform/Scripts/python.exe'
$env:DVC_NO_ANALYTICS='1';$env:XDG_CACHE_HOME=Join-Path $taskRoot '.runtime/dvc-cache'
$env:DVC_SITE_CACHE_DIR=Join-Path $taskRoot '.runtime/dvc-site-cache'
$env:DVC_SYSTEM_CONFIG_DIR=Join-Path $taskRoot '.tooling/dvc-system'
$env:DVC_GLOBAL_CONFIG_DIR=Join-Path $taskRoot '.tooling/dvc-global'
Push-Location -LiteralPath $taskRoot
try{
 if(-not(Test-Path -LiteralPath '.git')){git -c init.defaultBranch=main init .;if($LASTEXITCODE -ne 0){throw 'Git initialization failed'}}
 if(-not(Test-Path -LiteralPath '.dvc')){& $taskPython -m dvc init; if($LASTEXITCODE -ne 0){throw 'DVC initialization failed'}}
 & $taskPython -m dvc remote add -f -d seaweed s3://guanye/dvc
 & $taskPython -m dvc remote modify seaweed endpointurl http://127.0.0.1:18333
 if($LASTEXITCODE -ne 0){throw 'DVC S3 configuration failed'}
 Write-Host 'DVC remote uses environment credentials; no S3 secret was saved in the tracked configuration.'
}finally{Pop-Location}
