$ErrorActionPreference='Stop';$taskRoot=Split-Path $PSScriptRoot -Parent
. (Join-Path $PSScriptRoot 'docker_path.ps1')
$taskDocker=Get-GuanyeDocker
& $taskDocker compose --project-directory $taskRoot --profile '*' down
if($LASTEXITCODE -ne 0){throw 'Service stop failed.'}
# Named volumes are intentionally retained; never use down --volumes for normal shutdown.
$taskNginx=Get-ChildItem -LiteralPath (Join-Path $taskRoot '.tooling/nginx') -Filter nginx.exe -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
if($taskNginx -and (Test-Path -LiteralPath (Join-Path $taskRoot '.runtime/nginx.pid'))){& $taskNginx.FullName -s quit -p (($taskNginx.Directory.FullName -replace '\\','/')+'/') -c ((Join-Path $taskRoot '.runtime/nginx.conf') -replace '\\','/')}
