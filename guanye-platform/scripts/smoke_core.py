"""Live core integration check using explicitly temporary, isolated QA records."""
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import boto3
from botocore.config import Config
import httpx
import psycopg
from psycopg.rows import dict_row
import redis
from integrations.platform_bridge import read_config,token

def main():
    config=read_config(ROOT)
    owner='qa-'+uuid.uuid4().hex
    collector=owner+'-collector'
    outsider='qa-'+uuid.uuid4().hex
    project=None;task_ids=[]
    result={'scope':'Live PostgreSQL, Redis, SeaweedFS, Spring roles, Celery quality and Parquet audit; temporary QA data only.', 'checks':{},'cleanup':False}
    db_args=dict(host='127.0.0.1',port=15432,dbname='guanye',user='guanye',password=config['POSTGRES_PASSWORD'],connect_timeout=5,row_factory=dict_row)
    cache=redis.Redis(host='127.0.0.1',port=16379,password=config['REDIS_PASSWORD'],socket_timeout=5)
    store=boto3.client('s3',endpoint_url='http://127.0.0.1:18333',aws_access_key_id=config['S3_ACCESS_KEY'],aws_secret_access_key=config['S3_SECRET_KEY'],region_name='us-east-1',config=Config(signature_version='s3v4',s3={'addressing_style':'path'},connect_timeout=5,read_timeout=15,retries={'max_attempts':1}))
    bucket='guanye';prefix='users/'+owner+'/'
    client=httpx.Client(base_url=os.getenv('GUANYE_TEST_BUSINESS_URL','http://127.0.0.1:18081/v1/'),trust_env=False,timeout=15)
    def call(path,method='GET',data=None,user=owner,expected=200):
        response=client.request(method,path,json=data,headers={'Authorization':'Bearer '+token({'id':user,'name':'temporary integration check'},config['GUANYE_BRIDGE_SECRET'])})
        if response.status_code!=expected:raise RuntimeError(f'{method} {path}: expected {expected}, got {response.status_code}: '+response.text[:600])
        return response.json() if response.content else None
    def task(kind,sample,args=None):
        created=call('tasks','POST',{'project_id':project,'kind':kind,'args':{'sample_ids':[sample],**(args or {})}});task_ids.append(created['id'])
        deadline=time.monotonic()+(900 if os.getenv('GUANYE_CHECK_EXTENDED')=='qwen' else 300 if os.getenv('GUANYE_CHECK_RESEARCH')=='1' else 100)
        while time.monotonic()<deadline:
            row=call('tasks/'+created['id'])
            if row['status']=='complete':return row['result']
            if row['status']=='failed':raise RuntimeError(kind+' task failed: '+str(row['error']))
            time.sleep(2)
        raise RuntimeError(kind+' task did not complete before the validation deadline')
    try:
        assert client.get('projects').status_code==401
        result['checks']['anonymous_business_request']='401'
        with psycopg.connect(**db_args) as db:assert db.execute('SELECT 1 AS ok').fetchone()['ok']==1
        result['checks']['postgres']='pass'
        assert cache.ping();key='guanye:qa:'+owner;cache.set(key,'check',ex=60);assert cache.get(key)==b'check';cache.delete(key)
        result['checks']['redis_authenticated_read_write']='pass'
        store.head_bucket(Bucket=bucket)
        source=ROOT.parent/'test-images/tomato_healthy.jpg'
        if source.exists():
            data=source.read_bytes();result['fixture_scope']='public PlantVillage deployment fixture; not an accuracy evaluation'
        else:
            from PIL import Image
            fixture=Image.new('RGB',(256,256));fixture.putdata([((x*3)%256,(y*5)%256,((x+y)*7)%256) for y in range(256) for x in range(256)])
            buffer=io.BytesIO();fixture.save(buffer,format='JPEG');data=buffer.getvalue()
            result['fixture_scope']='synthetic technical quality/audit fixture; no plant or diagnostic accuracy claim'
        image_key=prefix+'source/temporary-leaf.jpg' 
        store.put_object(Bucket=bucket,Key=image_key,Body=data,ContentType='image/jpeg')
        assert store.get_object(Bucket=bucket,Key=image_key)['Body'].read()==data
        result['checks']['s3_authenticated_read_write']='pass'
        project=call('projects','POST',{'name':'临时接口验收（结束后清理）','crop':'番茄'})['id']
        call('projects/'+project+'/members','POST',{'user_id':collector,'role':'COLLECTOR'})
        sample=call('samples','POST',{'project_id':project,'source_id':'QA-LEAF','object_key':image_key,'sha256':hashlib.sha256(data).hexdigest(),'quality':{},'model_answer':''})['id']
        call('samples/'+sample,user=outsider,expected=404)
        call('samples/'+sample,user=collector)
        call('samples','POST',{'project_id':project,'source_id':'QA-INVALID','object_key':'users/another/not-owned.jpg','sha256':hashlib.sha256(data).hexdigest(),'quality':{},'model_answer':''},expected=400)
        device=call('devices','POST',{'project_id':project,'name':'临时接口验收设备（无实体）','kind':'camera'})['id']
        call('devices/'+device,'PATCH',{'status':'approved'},user=collector,expected=403)
        call('devices/'+device,'PATCH',{'status':'approved'})
        result['checks']['project_role_and_object_boundaries']='pass; outsider 404, wrong owner object 400, collector approval 403'
        quality=task('quality',sample)
        assert quality['rows'][0]['width']==256 and quality['rows'][0]['height']==256
        result['checks']['celery_quality']=quality
        audit=task('dataset_audit',sample)
        result['checks']['celery_dataset_audit']=audit
        def artifact_keys(v):
            if isinstance(v,str) and v.startswith(prefix):return [v]
            if isinstance(v,dict):return [k for x in v.values() for k in artifact_keys(x)]
            if isinstance(v,list):return [k for x in v for k in artifact_keys(x)]
            return []
        parquet=next(k for k in artifact_keys(audit) if k.endswith('.parquet'))
        import pyarrow.parquet as pq
        table=pq.read_table(io.BytesIO(store.get_object(Bucket=bucket,Key=parquet)['Body'].read()))
        assert table.num_rows==1
        result['checks']['parquet_readback_rows']=table.num_rows
        with psycopg.connect(**db_args) as db:
            rows=db.execute("SELECT event_type,dispatched,kafka_sent FROM outbox WHERE project_id=%s",(uuid.UUID(project),)).fetchall()
            assert any(r['event_type']=='task.created' and r['dispatched'] for r in rows)
            result['checks']['outbox_task_dispatch']='pass'
            result['checks']['kafka']='all observed outbox events delivered' if rows and all(r['kafka_sent'] for r in rows) else 'delivery pending or Kafka profile inactive; events retained for later delivery'
        if os.getenv('GUANYE_CHECK_RESEARCH')=='1':
            from smoke_research import check
            result['checks']['research']=check(config,project,sample,task,store,bucket,prefix)
            result['scope']+=' Includes actual queued public-model inference and MLflow tracking/artifact roundtrip.'
        if os.getenv('GUANYE_CHECK_EXTENDED'):
            from smoke_extended import check
            stage=os.environ['GUANYE_CHECK_EXTENDED']
            result['checks'][stage]=check(stage,config=config,project=project,device=device,owner=owner,db_args=db_args,task=task,sample=sample,store=store,bucket=bucket,prefix=prefix,call=call)
            result['scope']+=' Includes actual extended-stage service checks: '+stage+'.'
        result['status']='pass'
    except Exception as error:
        result['status']='failed';result['error']=str(error)
    finally:
        try:
            if project:
                with psycopg.connect(**db_args) as db:
                    row=db.execute('SELECT owner_id FROM projects WHERE id=%s',(uuid.UUID(project),)).fetchone()
                    if not row or row['owner_id']!=owner:raise RuntimeError('QA cleanup ownership guard failed')
                    db.execute('DELETE FROM reviews WHERE sample_id IN (SELECT id FROM platform_samples WHERE project_id=%s)',(uuid.UUID(project),))
                    for name in ['tasks','devices','model_experiments','dataset_versions','platform_samples','outbox','audit_log','memberships']:
                        db.execute('DELETE FROM '+name+' WHERE project_id=%s',(uuid.UUID(project),))
                    db.execute('DELETE FROM projects WHERE id=%s AND owner_id=%s',(uuid.UUID(project),owner))
            if project and os.getenv('GUANYE_CHECK_EXTENDED'):
                from smoke_extended import cleanup
                cleanup(os.environ['GUANYE_CHECK_EXTENDED'],project,config)
            objects=store.list_objects_v2(Bucket=bucket,Prefix=prefix).get('Contents',[])
            if objects:store.delete_objects(Bucket=bucket,Delete={'Objects':[{'Key':o['Key']} for o in objects]})
            for task_id in task_ids:cache.delete('celery-task-meta-'+task_id)
            result['cleanup']=True
        except Exception as error:result['cleanup_error']=str(error)
        client.close()
    # No JWTs, passwords, real accounts, or face records are exported.
    report_path=ROOT/('.runtime/research-verification.json' if os.getenv('GUANYE_CHECK_RESEARCH')=='1' else '.runtime/core-verification.json')
    if os.getenv('GUANYE_CHECK_EXTENDED'):report_path=ROOT/'.runtime/extended'/(os.environ['GUANYE_CHECK_EXTENDED']+'-verification.json')
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return 0 if result.get('status')=='pass' and result['cleanup'] else 1

if __name__=='__main__':sys.exit(main())
