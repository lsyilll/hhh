"""Scheduled Parquet snapshots, with project-scoped runs for manual verification."""
from datetime import datetime, timedelta, timezone
from airflow.sdk import dag, task


@dag(schedule='0 2 * * *',start_date=datetime(2026,1,1,tzinfo=timezone.utc),catchup=False,tags=['guanye','quality'],max_active_runs=1)
def daily_quality():
    @task(retries=2,retry_delay=timedelta(seconds=10),execution_timeout=timedelta(minutes=2))
    def snapshot():
        import hashlib
        import io
        import os
        import uuid
        from airflow.sdk import get_current_context
        import boto3
        import psycopg2
        import psycopg2.extras
        import pyarrow as pa
        import pyarrow.parquet as pq

        context=get_current_context();conf=context['dag_run'].conf or {}
        project=uuid.UUID(conf['project_id']) if conf.get('project_id') else None
        scope='offline/daily/'+datetime.now(timezone.utc).strftime('%Y-%m-%d')+'/'
        with psycopg2.connect(os.environ['DATABASE_DSN']) as db:
            with db.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                if project:
                    cursor.execute('SELECT owner_id FROM projects WHERE id=%s',(str(project),));row=cursor.fetchone()
                    if not row:raise ValueError('Requested project does not exist')
                    if conf.get('qa_scope'):
                        if not row['owner_id'].startswith('qa-'):raise ValueError('QA snapshot requires a temporary QA owner')
                        scope='users/'+row['owner_id']+'/offline/'
                    else:scope='offline/projects/'+str(project)+'/'
                query='SELECT id::text,project_id::text,owner_id,sha256,label,review_status,created::text FROM platform_samples'
                cursor.execute(query+(' WHERE project_id=%s' if project else '')+' ORDER BY id',((str(project),) if project else ()))
                rows=[dict(row) for row in cursor.fetchall()]
        if not rows:return {'rows':0,'note':'No data; no invented quality metrics'}
        buffer=io.BytesIO();pq.write_table(pa.Table.from_pylist(rows),buffer)
        digest=hashlib.sha256(buffer.getvalue()).hexdigest();key=scope+digest+'.parquet'
        client=boto3.client('s3',endpoint_url=os.environ['S3_ENDPOINT'])
        client.put_object(Bucket=os.getenv('S3_BUCKET','guanye'),Key=key,Body=buffer.getvalue())
        return {'rows':len(rows),'object_key':key,'sha256':digest,'scope':'data snapshot; labels and review states preserved'}
    snapshot()


daily_quality()
