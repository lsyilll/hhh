"""Actual queued model inference and MLflow checks; temporary QA records only."""
import io
import json
import os
import time
import uuid
from urllib.parse import urlsplit

import httpx
import numpy as np
from PIL import Image


def check(config,project,sample,task,store,bucket,prefix):
    result={}
    with httpx.Client(base_url='http://127.0.0.1:18082',trust_env=False,timeout=20) as models:
        response=models.get('/health');response.raise_for_status();health=response.json()
        assert all(health['weights'][n] for n in ('dinov2b','dinov2l','sam2'))
        assert health['device']=='cpu'
        payload={'task_id':str(uuid.uuid4()),'owner_id':'qa-untrusted','kind':'features','args':{},'samples':[{'id':sample}]}
        assert models.post('/run',json=payload).status_code==401
        result['model_authentication']='anonymous /run rejected with 401'
    for encoder,dimension in [('dinov2b',768),('dinov2l',1024)]:
        started=time.monotonic()
        row=task('features',sample,{'model':encoder})
        assert row['encoder']==encoder and row['count']==1 and row['dimension']==dimension
        assert row['features'].startswith(prefix) and row['faiss'].startswith(prefix)
        vector=np.load(io.BytesIO(store.get_object(Bucket=bucket,Key=row['features'])['Body'].read()),allow_pickle=False)
        assert vector.shape==(1,dimension) and np.isfinite(vector).all()
        assert abs(float(np.linalg.norm(vector[0]))-1)<1e-5
        assert len(store.get_object(Bucket=bucket,Key=row['faiss'])['Body'].read())>dimension*4
        result[encoder]={'queued_task':'pass','dimension':dimension,'norm':round(float(np.linalg.norm(vector[0])),6),'seconds':round(time.monotonic()-started,2)}
    started=time.monotonic()
    row=task('segment',sample,{'segmenter':'sam2','box':[0.5,0.5,0.9,0.9]})
    assert row['model']=='sam2' and row['prompt_type']=='box' and row['prompt'] is None
    candidates=row['samples'][0]['candidates'];assert candidates
    for candidate in candidates:
        assert candidate['mask'].startswith(prefix)
        data=store.get_object(Bucket=bucket,Key=candidate['mask'])['Body'].read()
        with Image.open(io.BytesIO(data)) as mask:
            assert mask.size==(256,256) and np.asarray(mask).any()
        assert 0<=candidate['segmentation_score']<=1
    result['sam2']={'queued_task':'pass','prompt_type':'box','masks':len(candidates),'mask_size':[256,256],'seconds':round(time.monotonic()-started,2)}
    # MLflow records a technical smoke run, never a fabricated disease accuracy.
    experiment_id=None;run_id=None;artifact_path=None
    with httpx.Client(base_url='http://127.0.0.1:15000',trust_env=False,timeout=30) as tracking:
        def api(path,data):
            response=tracking.post('/api/2.0/mlflow/'+path,json=data)
            response.raise_for_status()
            return response.json() if response.content else {}
        try:
            experiment_id=api('experiments/create',{'name':'guanye-qa-'+uuid.uuid4().hex,'tags':[{'key':'guanye.qa','value':'true'}]})['experiment_id']
            run=api('runs/create',{'experiment_id':experiment_id,'run_name':'temporary functionality check; not accuracy evaluation','start_time':int(time.time()*1000),'tags':[{'key':'guanye.qa','value':'true'}]})['run']
            run_id=run['info']['run_id']
            api('runs/log-parameter',{'run_id':run_id,'key':'encoder','value':'dinov2b'})
            api('runs/log-metric',{'run_id':run_id,'key':'feature_dimension','value':768,'timestamp':int(time.time()*1000),'step':0})
            uri=urlsplit(run['info']['artifact_uri'])
            if uri.scheme!='mlflow-artifacts' or uri.netloc:raise RuntimeError('Unexpected MLflow artifact location; will not write to it')
            artifact_path=uri.path.strip('/')+'/verification.json'
            if not artifact_path.startswith(str(experiment_id)+'/'+run_id+'/'):raise RuntimeError('MLflow QA artifact ownership guard failed')
            url='/api/2.0/mlflow-artifacts/artifacts/'+artifact_path
            content=json.dumps({'scope':'functionality only','models':result},sort_keys=True).encode()
            response=tracking.put(url,content=content,headers={'Content-Type':'application/json'});response.raise_for_status()
            response=tracking.get(url);response.raise_for_status();assert response.content==content
            api('runs/update',{'run_id':run_id,'status':'FINISHED','end_time':int(time.time()*1000)})
            response=tracking.get('/api/2.0/mlflow/runs/get',params={'run_id':run_id});response.raise_for_status()
            saved=response.json()['run'];assert saved['data']['metrics'][0]['value']==768
            result['mlflow']='pass: PostgreSQL run/parameter/metric and S3 artifact roundtrip'
        finally:
            if artifact_path:
                response=tracking.delete('/api/2.0/mlflow-artifacts/artifacts/'+artifact_path);response.raise_for_status()
            if run_id:api('runs/delete',{'run_id':run_id})
            if experiment_id:api('experiments/delete',{'experiment_id':experiment_id})
    result['mlflow_qa_cleanup']='test artifact deleted; temporary run and experiment soft-deleted through official API'
    return result


if __name__=='__main__':
    os.environ['GUANYE_CHECK_RESEARCH']='1'
    import smoke_core
    raise SystemExit(smoke_core.main())
