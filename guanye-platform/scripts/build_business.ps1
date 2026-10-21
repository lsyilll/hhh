$ErrorActionPreference='Stop'
$taskRoot=Split-Path $PSScriptRoot -Parent
$taskJava=Get-ChildItem -LiteralPath (Join-Path $taskRoot '.tooling') -Directory | Where-Object Name -Like 'jdk-21*' | Select-Object -First 1
if(-not $taskJava){throw 'Run the verified portable Java setup first.'}
$env:JAVA_HOME=$taskJava.FullName
$env:PATH="$($taskJava.FullName)\bin;C:\Windows\System32;$env:PATH"
$env:MAVEN_OPTS="-Duser.home=$taskRoot/.tooling/maven-home"
& (Join-Path $taskRoot '.tooling/apache-maven-3.9.11/bin/mvn.cmd') -B -s (Join-Path $taskRoot 'business/settings.xml') "-Dmaven.repo.local=$taskRoot/.tooling/m2" -f (Join-Path $taskRoot 'business/pom.xml') package
if($LASTEXITCODE -ne 0){throw 'Java build failed.'}
