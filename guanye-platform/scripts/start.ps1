param([ValidateSet('core','devices','analytics','warehouse','research','offline','monitoring','models','accelerator')][string[]]$Profiles=@('core'),[switch]$ExclusiveModelGpu)
$ErrorActionPreference='Stop';$taskRoot=Split-Path $PSScriptRoot -Parent
. (Join-Path $PSScriptRoot 'docker_path.ps1')
$taskDocker=Get-GuanyeDocker
$taskGitPath=Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd'
if((-not(Get-Command git -ErrorAction SilentlyContinue)) -and (Test-Path -LiteralPath (Join-Path $taskGitPath 'git.exe'))){$env:PATH="$taskGitPath;C:\Windows\System32;$env:PATH"}
if(-not(Test-Path -LiteralPath (Join-Path $taskRoot '.env'))){throw 'Run init_config.py first.'}
$taskBudget=3.0
foreach($taskProfile in ($Profiles | Select-Object -Unique)){switch($taskProfile){devices{$taskBudget+=0.4};analytics{$taskBudget+=3};warehouse{$taskBudget+=6};research{$taskBudget+=1};offline{$taskBudget+=2};monitoring{$taskBudget+=1.5};models{$taskBudget+=12};accelerator{$taskBudget+=2}}}
if(($Profiles -contains 'warehouse') -and -not($Profiles -contains 'analytics')){$taskBudget+=3}
if($taskBudget -gt 18){throw 'This profile combination exceeds the 18GB development VM budget. Start research/models and analytics/warehouse in separate sessions.'}
$taskArguments=@('compose','--project-directory',$taskRoot,'-f',(Join-Path $taskRoot 'compose.yaml'))
if($ExclusiveModelGpu){
  $taskProcessFile=Join-Path (Split-Path $taskRoot -Parent) 'logs/server-process.json'
  if(Test-Path -LiteralPath $taskProcessFile){throw 'Stop AgriChat before enabling exclusive model GPU mode. Its full model cannot share a 12GB GPU with the research models.'}
  $taskArguments+=@('-f',(Join-Path $taskRoot 'compose.gpu.yaml'))
}
if($Profiles -contains 'accelerator'){
 if(Test-Path -LiteralPath (Join-Path (Split-Path $taskRoot -Parent) 'logs/server-process.json')){throw 'Stop AgriChat before starting the GPU accelerator profile.'}
}
if($Profiles -contains 'monitoring'){$env:OTEL_ENABLED='1'}
foreach($taskProfile in $Profiles){if($taskProfile -ne 'core'){$taskArguments+=@('--profile',$taskProfile)}}
if($Profiles -contains 'warehouse'){$taskArguments+=@('--profile','analytics')}
& $taskDocker @taskArguments config --quiet
if($LASTEXITCODE -ne 0){throw 'Compose config validation failed.'}
$taskPython=Join-Path $taskRoot '.venv-platform/Scripts/python.exe'
if(-not(Test-Path -LiteralPath $taskPython)){throw 'The platform Python environment is missing. Run scripts/bootstrap_python.ps1 first.'}
& $taskPython (Join-Path $PSScriptRoot 'compose_run.py') --docker $taskDocker @taskArguments --progress plain up -d --build
if($LASTEXITCODE -ne 0){throw ('Service startup failed. Complete Docker output is saved in: '+(Join-Path $taskRoot '.runtime/compose-last-start.log'))}
