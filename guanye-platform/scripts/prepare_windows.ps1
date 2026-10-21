#Requires -RunAsAdministrator
$ErrorActionPreference='Stop'
Write-Host 'This installs/enables WSL2. Windows may require a restart. Existing distributions are not deleted.'
& "$env:SystemRoot\System32\wsl.exe" --install --no-distribution
if($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 3010){throw 'WSL setup did not complete; inspect Windows output.'}
Write-Host 'Complete any Windows restart yourself. Install Docker Desktop from the official page and accept its terms yourself:'
Write-Host 'https://docs.docker.com/desktop/setup/install/windows-install/'
Write-Host 'Then launch Docker Desktop using its WSL2 backend. This script does not accept a license or reboot Windows.'
