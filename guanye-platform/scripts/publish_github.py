"""Publish only the reviewed source commit to the user-selected repository."""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
import urllib.parse
import urllib.error
from network_env import windows_download_env

ROOT=Path(__file__).resolve().parents[1]
REPOSITORY='lsyilll/hhh'
BRANCH='codex/guanye-full-stack'

def main():
    configuration=json.loads((ROOT/'.runtime/github-upload.json').read_text(encoding='utf-8'))
    checkout=Path(configuration['checkout']).resolve()
    git=Path(os.environ['USERPROFILE'])/'.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe'
    env,_=windows_download_env()
    for name in ('GCM_INTERACTIVE','GIT_TERMINAL_PROMPT'):env.pop(name,None)
    env['PATH']=str(git.parent)+os.pathsep+str(Path(os.environ.get('SystemRoot',r'C:\Windows'))/'System32')+os.pathsep+env.get('PATH','')
    state={'repository':REPOSITORY,'branch':BRANCH,'status':'running','started_at':datetime.now().isoformat()}
    output=ROOT/'.runtime/github-publish.json'
    def capture(*args):
        return subprocess.check_output([str(git),'-C',str(checkout),*args],env=env,stderr=subprocess.PIPE).decode('utf-8').strip()
    def api(path,body=None,method=None):
        request=urllib.request.Request('https://api.github.com/'+path,headers=headers,data=json.dumps(body).encode() if body is not None else None,method=method)
        with opener.open(request,timeout=45) as response:return json.load(response)
    try:
        assert configuration['repository']==REPOSITORY
        assert capture('remote','get-url','origin')=='https://github.com/'+REPOSITORY+'.git'
        assert capture('branch','--show-current')==BRANCH
        assert capture('rev-parse','HEAD')==configuration['commit']
        if capture('status','--porcelain'):raise RuntimeError('The reviewed checkout has new changes. Review them before publishing.')
        changed=capture('diff','--name-only','origin/main...HEAD').splitlines()
        if not changed or any(not path.startswith('guanye-platform/') and path!='.github/workflows/guanye-platform.yml' for path in changed):raise RuntimeError('Commit contains changes outside the reviewed source scope')
        print('Publishing reviewed commit '+configuration['commit'][:12]+' to '+REPOSITORY+' / '+BRANCH,flush=True)
        print('Git may ask you to sign in through its normal GitHub browser flow.',flush=True)
        subprocess.run([str(git),'-C',str(checkout),'push','--set-upstream','origin',BRANCH],env=env,check=True)
        result=subprocess.run([str(git),'-C',str(checkout),'credential','fill'],input=b'protocol=https\nhost=github.com\npath=lsyilll/hhh.git\n\n',stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,check=True)
        credential=dict(line.split('=',1) for line in result.stdout.decode().splitlines() if '=' in line)
        if not credential.get('password'):raise RuntimeError('GitHub sign-in did not return an API credential; the branch was pushed but PR creation still needs sign-in')
        headers={'Accept':'application/vnd.github+json','Content-Type':'application/json','X-GitHub-Api-Version':'2022-11-28','User-Agent':'Guanye-platform-deployment','Authorization':'Bearer '+credential['password']}
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({'https':env['HTTPS_PROXY']})) if env.get('HTTPS_PROXY') else urllib.request.build_opener()
        existing=api('repos/'+REPOSITORY+'/pulls?state=open&head='+urllib.parse.quote('lsyilll:'+BRANCH,safe=''))
        if existing:pr=existing[0]
        else:
            body='''Add an independent Guanye research and collection platform under `guanye-platform/`, preserving the existing tutorial files. The platform connects project roles, queued image tasks, public-model alternatives, device communication, versioned datasets, streaming analytics, offline snapshots and observability.

Local validation: 27 Python tests, 7 JUnit tests, 3 Playwright browser tests and a production frontend build. Actual service checks cover PostgreSQL/Redis/S3/Spring/Celery, DINOv2 B/L and SAM 2.1, FAISS/MLflow, MQTT 5 mTLS, OpenTelemetry/Prometheus/Grafana, Kafka/Flink SQL/Doris, Airflow/Parquet, DVC/Git and Nginx TLS/mTLS. A numeric-head TensorRT check passed on an RTX 3060 against ONNX Runtime; this does not export or accelerate the full AgriChat model.

Qwen3.5-4B generated an image-grounded report through the actual task queue; the 96-token technical check was truncated. Original gated DINOv3/SAM3 weights, physical camera/microphone/Raspberry Pi/GPIO tests and disease-accuracy evaluation remain external prerequisites. Heavy services switch modes within the local RAM budget.

No model weights, user photos, face/audio records, runtime databases, private configuration, credentials, SDKs or build caches are included. The workflow adds Python/Java/frontend checks, Compose validation and an ephemeral core-service integration test. Remote CI results are pending this PR. No merge is performed.'''
            pr=api('repos/'+REPOSITORY+'/pulls',{'title':'Add Guanye research platform and full-stack CI','head':BRANCH,'base':'main','draft':True,'body':body})
        state.update(status='published',commit=configuration['commit'],pull_request=pr['html_url'],pull_request_number=pr['number'],draft=pr['draft'],finished_at=datetime.now().isoformat())
        print('Draft pull request: '+pr['html_url'],flush=True)
        return 0
    except Exception as error:
        state.update(status='failed',error=type(error).__name__+': '+str(error),finished_at=datetime.now().isoformat())
        print('Publish failed: '+state['error'],flush=True);return 1
    finally:output.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':raise SystemExit(main())
