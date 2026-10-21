"""Durable, idempotent batch execution; database status is authoritative."""
import io,json,os,time,uuid,functools
from datetime import datetime,timezone
from pathlib import Path
import httpx,numpy as np
from celery import Celery
from celery.exceptions import SoftTimeLimitExceeded
from PIL import Image,ImageOps
from worker.common import connection,s3,bucket,ensure_bucket,event
from worker.observability import observed

app=Celery('guanye',broker=os.environ['REDIS_URL'],backend=os.environ['REDIS_URL'])
app.conf.update(task_acks_late=True,task_reject_on_worker_lost=True,worker_prefetch_multiplier=1,
    broker_transport_options={'visibility_timeout':3600},task_soft_time_limit=1800,task_time_limit=1860,
    result_expires=86400,task_serializer='json',accept_content=['json'])

def quality(data):
    import cv2
    with Image.open(io.BytesIO(data)) as src:
        image=ImageOps.exif_transpose(src).convert('RGB');width,height=image.size;image.thumbnail((1024,1024));gray=cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2GRAY)
    sharp=float(cv2.Laplacian(gray,cv2.CV_64F).var());brightness=float(gray.mean());codes=[]
    if min(width,height)<384:codes.append('resolution')
    if sharp<55:codes.append('blur')
    if brightness<45:codes.append('dark')
    if brightness>215:codes.append('bright')
    return {'width':width,'height':height,'sharpness':round(sharp,2),'brightness':round(brightness,1),'issues':codes,'method':'OpenCV advisory checks'}

def samples_for(task):
    ids=task['args'].get('sample_ids',[])
    with connection() as db:
        if ids:
            rows=db.execute('SELECT * FROM platform_samples WHERE project_id=%s AND id=ANY(%s::uuid[])',(task['project_id'],ids)).fetchall()
            if len(rows)!=len(set(ids)):raise ValueError('Selected sample does not belong to this project')
        else:rows=db.execute('SELECT * FROM platform_samples WHERE project_id=%s ORDER BY created LIMIT 100',(task['project_id'],)).fetchall()
    return rows

def artifact(task,name,data,content_type):
    key=f'users/{task["owner_id"]}/tasks/{task["id"]}/{name}'
    s3().put_object(Bucket=bucket(),Key=key,Body=data,ContentType=content_type)
    return key

def call_model(task,rows):
    response=httpx.post(os.environ['MODEL_URL']+'/run',json={'task_id':str(task['id']),'owner_id':task['owner_id'],
        'kind':task['kind'],'args':task['args'],'samples':[{key:str(row[key]) if key in ('id','project_id') else row[key] for key in ('id','project_id','owner_id','object_key','label','review_status','model_answer','quality','notes')} for row in rows]},
        headers={'Authorization':'Bearer '+os.environ['MODEL_SERVICE_SECRET']},timeout=1700,trust_env=False)
    response.raise_for_status();return response.json()

def leased(function):
    @functools.wraps(function)
    def wrapped(self,task_id):
        with connection() as lease:
            acquired=lease.execute('SELECT pg_try_advisory_lock(hashtextextended(%s,0)) AS locked',(task_id,)).fetchone()['locked'];lease.commit()
            if not acquired:return {'status':'running','idempotent':True}
            try:return function(self,task_id)
            finally:lease.execute('SELECT pg_advisory_unlock(hashtextextended(%s,0))',(task_id,));lease.commit()
    return wrapped

@app.task(bind=True,name='guanye.run',max_retries=2)
@observed
@leased
def run(self,task_id):
    with connection() as db:
        task=db.execute('SELECT * FROM tasks WHERE id=%s FOR UPDATE',(uuid.UUID(task_id),)).fetchone()
        if not task:raise ValueError('Unknown task')
        if task['status'] in ('complete','cancelled'):return {'status':task['status'],'idempotent':True}
        db.execute("UPDATE tasks SET status='running',started=now(),attempts=attempts+1,error=NULL WHERE id=%s",(task_id,));event(db,task,'task.started')
    started=time.monotonic()
    try:
        ensure_bucket();rows=samples_for(task)
        if not rows:raise ValueError('Project contains no selected samples')
        if task['kind']=='quality':
            output=[]
            for index,row in enumerate(rows):
                data=s3().get_object(Bucket=bucket(),Key=row['object_key'])['Body'].read();q=quality(data)
                with connection() as db:
                    db.execute('UPDATE platform_samples SET quality=%s::jsonb WHERE id=%s',(json.dumps(q),row['id']))
                    db.execute('UPDATE tasks SET progress=%s WHERE id=%s',(int((index+1)*100/len(rows)),task_id))
                    event(db,task,'sample.quality',sample_id=str(row['id']),quality_codes=q['issues'])
                output.append({'sample_id':str(row['id']),**q})
            result={'rows':output}
        elif task['kind']=='dataset_audit':
            import hashlib,pyarrow as pa,pyarrow.parquet as pq
            manifest=[];seen={}
            for row in rows:
                data=s3().get_object(Bucket=bucket(),Key=row['object_key'])['Body'].read();sha=hashlib.sha256(data).hexdigest()
                try:q=quality(data);valid=True
                except Exception:q={};valid=False
                manifest.append({'id':str(row['id']),'object_key':row['object_key'],'sha256':sha,'label':row['label'],
                    'confirmed':row['review_status']=='confirmed','valid_image':valid,'duplicate_of':seen.get(sha,''),'width':q.get('width',0),'height':q.get('height',0)})
                seen.setdefault(sha,str(row['id']))
            raw='\n'.join(json.dumps(r,ensure_ascii=False,sort_keys=True) for r in manifest).encode();digest=hashlib.sha256(raw).hexdigest()
            key=artifact(task,'manifest.jsonl',raw,'application/x-ndjson');buffer=io.BytesIO();pq.write_table(pa.Table.from_pylist(manifest),buffer)
            parquet=artifact(task,'manifest.parquet',buffer.getvalue(),'application/vnd.apache.parquet')
            with connection() as db:db.execute('INSERT INTO dataset_versions VALUES(%s,%s,%s,%s,%s,%s,now())',(uuid.uuid4(),task['project_id'],task['owner_id'],digest,key,len(manifest)))
            result={'manifest':key,'parquet':parquet,'version':digest,'rows':len(manifest),'duplicates':sum(bool(r['duplicate_of']) for r in manifest),'corrupt':sum(not r['valid_image'] for r in manifest)}
        else:result=call_model(task,rows)
        with connection() as db:
            db.execute("UPDATE tasks SET status='complete',progress=100,result=%s::jsonb,finished=now() WHERE id=%s",(json.dumps(result,ensure_ascii=False),task_id))
            event(db,task,'task.complete',duration_ms=round((time.monotonic()-started)*1000),status='complete')
        return result
    except Exception as error:
        # Keep detailed failures in local worker logs, never insert credentials/URLs into the browser response.
        import logging;logging.getLogger('guanye.worker').exception('Task failed: %s',task_id)
        retryable=isinstance(error,(httpx.TransportError,ConnectionError)) and self.request.retries<self.max_retries
        with connection() as db:
            db.execute('UPDATE tasks SET status=%s,error=%s,finished=now() WHERE id=%s',('queued' if retryable else 'failed',type(error).__name__+': '+str(error)[:500],task_id))
            event(db,task,'task.failed',status='failed',duration_ms=round((time.monotonic()-started)*1000))
        if retryable:raise self.retry(exc=error,countdown=10*(self.request.retries+1))
        raise
