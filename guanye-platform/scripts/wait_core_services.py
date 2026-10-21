"""Wait for actual authenticated dependencies, including initialized object storage."""
import argparse
import os
from pathlib import Path
import sys
import time
import boto3
from botocore.config import Config
import httpx
import psycopg
import redis

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from integrations.platform_bridge import read_config

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--timeout',type=int,default=180);args=parser.parse_args()
    config=read_config(ROOT)
    store=boto3.client('s3',endpoint_url='http://127.0.0.1:18333',aws_access_key_id=config['S3_ACCESS_KEY'],aws_secret_access_key=config['S3_SECRET_KEY'],region_name='us-east-1',config=Config(signature_version='s3v4',s3={'addressing_style':'path'},proxies={},connect_timeout=2,read_timeout=2,retries={'max_attempts':0}))
    cache=redis.Redis(host='127.0.0.1',port=16379,password=config['REDIS_PASSWORD'],socket_timeout=2,socket_connect_timeout=2)
    deadline=time.monotonic()+args.timeout;last='not checked'
    try:
        while time.monotonic()<deadline:
            service='business'
            try:
                response=httpx.get('http://127.0.0.1:18081/actuator/health',trust_env=False,timeout=2);response.raise_for_status()
                service='postgres'
                with psycopg.connect(host='127.0.0.1',port=15432,dbname='guanye',user='guanye',password=config['POSTGRES_PASSWORD'],connect_timeout=2) as db:
                    assert db.execute('SELECT 1').fetchone()[0]==1
                service='redis';assert cache.ping()
                service='S3 initialized bucket';store.head_bucket(Bucket='guanye')
                print('Core ready: authenticated PostgreSQL / Redis / S3 bucket and Spring health',flush=True)
                return 0
            except Exception as error:
                current=service+': '+type(error).__name__
                if current!=last:print('Waiting for '+current,flush=True)
                last=current;time.sleep(2)
        raise RuntimeError('Core readiness deadline exceeded: '+last)
    finally:cache.close();store.close()

if __name__=='__main__':raise SystemExit(main())
