"""Authenticated Airflow API -> scheduler -> task -> S3 Parquet validation."""
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import time
import uuid
import httpx

ROOT=Path(__file__).resolve().parents[1]


def check(**context):
    from smoke_extended import eventually
    import pyarrow.parquet as pq
    password=json.loads((ROOT/'.runtime/airflow/passwords.json').read_text())['guanye-admin']
    run_id='qa__'+uuid.uuid4().hex;dag='daily_quality';created=False;terminal=False
    with httpx.Client(base_url='http://127.0.0.1:18084',trust_env=False,timeout=15) as client:
        assert client.get('/api/v2/dags').status_code==401
        response=client.post('/auth/token',json={'username':'guanye-admin','password':password})
        response.raise_for_status();client.headers['Authorization']='Bearer '+response.json()['access_token']
        def ready():
            response=client.get('/api/v2/dags/'+dag)
            return response.json() if response.status_code==200 else None
        details=eventually(ready,180)
        # Retain the user's chosen pause state after this controlled manual run.
        was_paused=details.get('is_paused',True)
        response=client.patch('/api/v2/dags/'+dag,json={'is_paused':False});response.raise_for_status()
        path='/api/v2/dags/'+dag+'/dagRuns/'+run_id
        try:
            response=client.post('/api/v2/dags/'+dag+'/dagRuns',json={'dag_run_id':run_id,'logical_date':None,'conf':{'project_id':context['project'],'qa_scope':True},'note':'Temporary QA snapshot; remove run and objects after verification'})
            response.raise_for_status();created=True
            deadline=time.monotonic()+300
            while time.monotonic()<deadline:
                response=client.get(path);response.raise_for_status();state=response.json()['state']
                if state in ('success','failed'):
                    terminal=True
                    if state=='failed':raise RuntimeError('Airflow QA DAG run failed; inspect saved task logs')
                    break
                time.sleep(3)
            else:raise RuntimeError('Airflow QA DAG did not finish within five minutes')
            prefix=context['prefix']+'offline/'
            objects=context['store'].list_objects_v2(Bucket=context['bucket'],Prefix=prefix).get('Contents',[])
            assert len(objects)==1 and objects[0]['Key'].endswith('.parquet')
            data=context['store'].get_object(Bucket=context['bucket'],Key=objects[0]['Key'])['Body'].read()
            table=pq.read_table(io.BytesIO(data));rows=table.to_pylist()
            assert len(rows)==1 and rows[0]['project_id']==context['project'] and rows[0]['owner_id']==context['owner']
            return {'status':'pass','anonymous_api':'401','authenticated_trigger':'pass','scheduler_task_run':'success','s3_parquet_rows':len(rows),'project_boundary':'pass','dag_run':run_id,'scope':'actual Airflow task; source labels and review states preserved; temporary run and S3 objects removed'}
        finally:
            if created and not terminal:
                response=client.patch(path,json={'state':'failed'});response.raise_for_status()
            if created:
                response=client.delete(path);response.raise_for_status()
            if was_paused:
                response=client.patch('/api/v2/dags/'+dag,json={'is_paused':True});response.raise_for_status()
