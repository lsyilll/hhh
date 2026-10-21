"""Build and test the console in a dedicated, disposable Playwright browser."""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import httpx
from network_env import windows_download_env

ROOT=Path(__file__).resolve().parents[1]

def main():
    node=Path(os.environ.get('USERPROFILE',''))/'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
    if not node.exists():
        import shutil
        node=Path(shutil.which('node') or '')
    if not node.is_file():raise RuntimeError('Node.js is required')
    env,_=windows_download_env();runtime=ROOT/'.runtime/extended';runtime.mkdir(parents=True,exist_ok=True)
    env.update(PLAYWRIGHT_BROWSERS_PATH=str(ROOT/'.runtime/playwright'),PLAYWRIGHT_DOWNLOAD_CONNECTION_TIMEOUT='120000',PLAYWRIGHT_JSON_OUTPUT_NAME=str(runtime/'playwright-results.json'))
    env['PATH']=str(node.parent)+os.pathsep+str(Path(os.environ.get('SystemRoot',r'C:\Windows'))/'System32')+os.pathsep+env.get('PATH','')
    env['GUANYE_TEST_EXTERNAL_SERVER']='1'
    report={'status':'running','started_at':datetime.now().isoformat(),'scope':'isolated Chromium; mocked HTTP boundary tests; no real browser profile or inference tasks'}
    target=runtime/'browser-verification.json'
    try:
        preview=None
        with (runtime/'browser.log').open('wb') as log:
            for args in [
                ['node_modules/typescript/bin/tsc','--noEmit'],
                ['node_modules/vite/bin/vite.js','build','--config','vite.config.mjs','--configLoader','native'],
                ['node_modules/@playwright/test/cli.js','install','chromium','--only-shell'],
                ['node_modules/@playwright/test/cli.js','test','--workers','1','--reporter','list,json'],
            ]:
                report['step']=args[0]+' '+args[1];target.write_text(json.dumps(report,indent=2))
                print('Browser verification: '+report['step'],flush=True)
                if args[1]=='test':
                    preview=subprocess.Popen([str(node),'node_modules/vite/bin/vite.js','preview','--host','127.0.0.1','--port','4173','--strictPort','--config','vite.config.mjs','--configLoader','native'],cwd=ROOT/'web',env=env,stdout=log,stderr=subprocess.STDOUT)
                    deadline=time.monotonic()+30
                    while True:
                        try:
                            if httpx.get('http://127.0.0.1:4173/platform/',trust_env=False,timeout=2).status_code==200:break
                        except httpx.HTTPError:pass
                        if preview.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Owned Vite preview failed to start')
                        time.sleep(.25)
                try:result=subprocess.run([str(node),*args],cwd=ROOT/'web',env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
                finally:
                    if preview is not None and preview.poll() is None:
                        preview.terminate();preview.wait(timeout=10)
                if result.returncode:raise RuntimeError(report['step']+' failed; see .runtime/extended/browser.log')
        data=json.loads((runtime/'playwright-results.json').read_text(encoding='utf-8'))
        assert data['stats']['unexpected']==0 and data['stats']['expected']>=3
        report.update(status='pass',tests=data['stats']['expected'],finished_at=datetime.now().isoformat())
        return 0
    except Exception as error:
        report.update(status='failed',error=str(error));print(str(error),flush=True);return 1
    finally:target.write_text(json.dumps(report,indent=2),encoding='utf-8')

if __name__=='__main__':raise SystemExit(main())
