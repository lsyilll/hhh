$ErrorActionPreference='Stop'
$taskRoot=Split-Path $PSScriptRoot -Parent
$taskNode='C:\Users\12203\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe'
$taskPnpm='C:\Users\12203\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules\pnpm\bin\pnpm.cjs'
Push-Location -LiteralPath (Join-Path $taskRoot 'web')
try {
 & $taskNode node_modules/typescript/bin/tsc --noEmit
 if($LASTEXITCODE -ne 0){throw 'TypeScript check failed.'}
 & $taskNode node_modules/vite/bin/vite.js build --config vite.config.mjs --configLoader native
 if($LASTEXITCODE -ne 0){throw 'Vite build failed.'}
}finally{Pop-Location}
