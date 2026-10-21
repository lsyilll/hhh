$ErrorActionPreference='Stop';$taskRoot=Split-Path $PSScriptRoot -Parent
$taskNginx=Get-ChildItem -LiteralPath (Join-Path $taskRoot '.tooling/nginx') -Filter nginx.exe -Recurse | Select-Object -First 1
if(-not $taskNginx){throw 'Download the official portable Nginx tool first.'}
$taskPrefix=($taskNginx.Directory.FullName -replace '\\','/')+'/'
$taskConfig=(Join-Path $taskRoot '.runtime/nginx.conf') -replace '\\','/'
& $taskNginx.FullName -t -p $taskPrefix -c $taskConfig
if($LASTEXITCODE -ne 0){throw 'Nginx configuration check failed.'}
Start-Process -FilePath $taskNginx.FullName -ArgumentList @('-p',$taskPrefix,'-c',$taskConfig) -WorkingDirectory $taskNginx.Directory.FullName -WindowStyle Hidden
Write-Host 'Local HTTPS gateway started at https://localhost:8443. Certificate trust must be handled by you; never bypass a browser security warning.'
