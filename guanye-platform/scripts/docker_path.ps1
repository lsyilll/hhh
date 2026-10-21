function Get-GuanyeDocker {
    $taskCommand=Get-Command docker -ErrorAction SilentlyContinue
    if($taskCommand){return $taskCommand.Source}
    $taskCandidates=@(
        (Join-Path $env:USERPROFILE 'AppData/Local/Programs/DockerDesktop/resources/bin/docker.exe'),
        'C:\Program Files\Docker\Docker\resources\bin\docker.exe'
    )
    foreach($taskCandidate in $taskCandidates){if(Test-Path -LiteralPath $taskCandidate -PathType Leaf){return $taskCandidate}}
    throw 'Docker Desktop executable was not found. Install it for this user or for all users.'
}
