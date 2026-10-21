"""Real Git / DVC / S3 roundtrip, confined to a disposable QA workspace."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]


def check(**context):
    from PIL import Image
    project=context['project'];prefix=context['prefix'];config=context['config']
    store=context['store'];bucket=context['bucket']
    # A second, synthetic technical fixture is needed to test non-overlapping splits.
    image=Image.new('RGB',(256,256));image.putdata([((x*7)%256,(y*11)%256,((x+y)*3)%256) for y in range(256) for x in range(256)])
    buffer=io.BytesIO();image.save(buffer,format='PNG');data=buffer.getvalue()
    key=prefix+'source/dvc-synthetic-fixture.png'
    store.put_object(Bucket=bucket,Key=key,Body=data,ContentType='image/png')
    context['call']('samples','POST',{'project_id':project,'source_id':'QA-SYNTHETIC-DVC','object_key':key,'sha256':hashlib.sha256(data).hexdigest(),'quality':{},'model_answer':''})
    runtime=ROOT/'.runtime/extended';runtime.mkdir(parents=True,exist_ok=True)
    git=shutil.which('git')
    if not git:
        candidate=Path(os.environ['USERPROFILE'])/'.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe'
        if candidate.is_file():git=str(candidate)
    if not git:raise RuntimeError('Git executable is required for the DVC roundtrip')
    env=os.environ.copy()
    env.update(PYTHONIOENCODING='utf-8',GUANYE_PLATFORM_ROOT=str(ROOT),DVC_NO_ANALYTICS='1',AWS_ACCESS_KEY_ID=config['S3_ACCESS_KEY'],AWS_SECRET_ACCESS_KEY=config['S3_SECRET_KEY'],AWS_DEFAULT_REGION='us-east-1',GIT_TERMINAL_PROMPT='0')
    env['PATH']=str(Path(sys.executable).parent)+os.pathsep+str(Path(git).parent)+os.pathsep+env.get('PATH','')
    with tempfile.TemporaryDirectory(prefix='dvc-qa-',dir=runtime) as temporary:
        folder=Path(temporary).resolve();assert folder.is_relative_to(runtime.resolve())
        env.update(XDG_CACHE_HOME=str(folder/'shared-cache'),DVC_SITE_CACHE_DIR=str(folder/'site-cache'),DVC_SYSTEM_CONFIG_DIR=str(folder/'system-config'),DVC_GLOBAL_CONFIG_DIR=str(folder/'global-config'))
        def run(where,args):
            result=subprocess.run(args,cwd=where,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
            output=result.stdout.decode('utf-8',errors='replace')
            if result.returncode:raise RuntimeError('DVC/Git check failed: '+output[-2400:])
            return output
        def dvc(where,*args):return run(where,[sys.executable,'-m','dvc',*args])
        first=folder/'producer';first.mkdir();(first/'scripts').mkdir()
        shutil.copyfile(ROOT/'scripts/dataset_pipeline.py',first/'scripts/dataset_pipeline.py')
        shutil.copyfile(ROOT/'dvc.yaml',first/'dvc.yaml')
        (first/'params.yaml').write_text('project_id: '+project+'\nseed: 42\n',encoding='utf-8')
        run(first,[git,'init','-b','main']);dvc(first,'init')
        remote='s3://'+bucket+'/'+prefix+'dvc-cache'
        dvc(first,'remote','add','-d','qa-seaweed',remote)
        dvc(first,'remote','modify','qa-seaweed','endpointurl','http://127.0.0.1:18333')
        dvc(first,'repro');dvc(first,'push')
        manifest=(first/'datasets/manifest.jsonl').read_bytes()
        audit=json.loads((first/'datasets/audit.json').read_text())
        splits=json.loads((first/'datasets/splits.json').read_text())
        assert audit['samples']==2 and audit['confirmed']==0
        assert len(splits['train'])==1 and len(splits['test'])==1 and not set(splits['train'])&set(splits['test'])
        run(first,[git,'add','.'])
        run(first,[git,'-c','user.name=Guanye QA','-c','user.email=qa@localhost','commit','-m','Record isolated QA dataset pipeline and lock'])
        commit=run(first,[git,'rev-parse','HEAD']).strip()
        second=folder/'consumer';run(first,[git,'worktree','add','--detach',str(second),'HEAD'])
        # This fresh Git checkout has no DVC cache or generated datasets before pull.
        assert not (second/'datasets/manifest.jsonl').exists()
        dvc(second,'pull')
        assert (second/'datasets/manifest.jsonl').read_bytes()==manifest
        assert json.loads((second/'datasets/splits.json').read_text())==splits
        assert (second/'.dvc/cache').resolve()!=(first/'.dvc/cache').resolve()
        objects=store.list_objects_v2(Bucket=bucket,Prefix=prefix+'dvc-cache/').get('Contents',[])
        assert objects
        return {'status':'pass','git_commit':commit,'dvc_pipeline_stages':['snapshot','audit','split'],'s3_cached_objects':len(objects),'fresh_checkout_pull':'identical manifest and splits restored from S3 with an empty local cache','manifest_sha256':hashlib.sha256(manifest).hexdigest(),'samples':2,'confirmed_labels':0,'split_overlap':0,'scope':'one public deployment fixture and one synthetic technical fixture; no disease accuracy claim; temporary Git repositories and S3 prefix cleaned'}
