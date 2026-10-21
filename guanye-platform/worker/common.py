import json,os,uuid
from contextlib import contextmanager
import boto3,psycopg
from psycopg.rows import dict_row
from botocore.config import Config

@contextmanager
def connection():
    with psycopg.connect(os.environ['DATABASE_DSN'],row_factory=dict_row) as db:yield db

def s3():
    return boto3.client('s3',endpoint_url=os.environ['S3_ENDPOINT'],config=Config(signature_version='s3v4',s3={'addressing_style':'path'}))

def bucket():return os.getenv('S3_BUCKET','guanye')

def ensure_bucket():
    from botocore.exceptions import ClientError
    try:s3().head_bucket(Bucket=bucket())
    except ClientError as error:
        if str(error.response['Error']['Code']) not in ('404','NoSuchBucket'):raise
        s3().create_bucket(Bucket=bucket())

def event(db,task,event_type,**payload):
    db.execute('INSERT INTO outbox(id,event_type,owner_id,project_id,payload) VALUES(%s,%s,%s,%s,%s::jsonb)',
        (uuid.uuid4(),event_type,task['owner_id'],task['project_id'],json.dumps({'task_id':str(task['id']),'kind':task['kind'],**payload},ensure_ascii=False)))
