"""User-launched, bounded service deployment; never changes global system settings."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import httpx
from network_env import windows_download_env

ROOT=Path(__file__).resolve().parents[1]
RUNTIME=ROOT/'.runtime/extended'


def docker_path():
    candidates=[shutil.which('docker'),str(Path(os.environ.get('USERPROFILE',''))/'AppData/Local/Programs/DockerDesktop/resources/bin/docker.exe'),r'C:\Program Files\Docker\Docker\resources\bin\docker.exe']
    for candidate in candidates:
        if candidate and Path(candidate).is_file():return candidate
    raise RuntimeError('Docker Desktop executable was not found')


class Runner:
    def __init__(self,stage):
        RUNTIME.mkdir(parents=True,exist_ok=True)
        self.stage=stage;self.docker=docker_path();self.env,proxy=windows_download_env()
        self.env['PYTHONIOENCODING']='utf-8'
        self.folder=RUNTIME/(stage+'-'+datetime.now().strftime('%Y%m%d-%H%M%S'));self.folder.mkdir()
        self.log=(self.folder/'docker.log').open('wb')
        self.state={'stage':stage,'status':'running','log':str(self.folder/'docker.log'),'started_at':datetime.now().isoformat()}
        self.save('initializing')

    def save(self,step):
        self.state.update(step=step,updated_at=datetime.now().isoformat())
        text=json.dumps(self.state,ensure_ascii=False,indent=2)
        (RUNTIME/(self.stage+'-driver.json')).write_text(text,encoding='utf-8')
        (self.folder/'status.json').write_text(text,encoding='utf-8')

    def run(self,arguments,label,capture=False):
        self.save(label);print(label,flush=True)
        self.log.write(('\n'+label+'\n').encode());self.log.flush()
        process=subprocess.Popen(arguments,cwd=ROOT,env=self.env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        chunks=[]
        for block in iter(lambda:process.stdout.read1(8192),b''):
            self.log.write(block);self.log.flush()
            if capture:chunks.append(block)
            else:sys.stdout.buffer.write(block);sys.stdout.buffer.flush()
        code=process.wait()
        if code:raise RuntimeError(label+' failed with exit code '+str(code)+'; full output is in '+str(self.folder/'docker.log'))
        return b''.join(chunks).decode('utf-8',errors='replace')

    def compose(self,*arguments,capture=False):
        common=[self.docker,'compose','--project-directory',str(ROOT),'-f',str(ROOT/'compose.yaml'),'-f',str(ROOT/'compose.extended.yaml'),'--profile','monitoring','--profile','devices']
        if self.stage in ('analytics','analytics_verify','offline','qwen'):common+=['--profile','analytics','--profile','warehouse','--profile','models','--profile','research','--profile','offline']
        return self.run(common+list(arguments),'Docker Compose: '+' '.join(arguments),capture)

    def healthy(self,url,seconds=120):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            try:
                response=httpx.get(url,trust_env=False,timeout=3)
                if response.status_code==200:return
            except httpx.HTTPError:pass
            time.sleep(3)
        raise RuntimeError('Service did not become healthy: '+url)

    def observability(self):
        self.healthy('http://127.0.0.1:18081/actuator/health',15)
        self.compose('config','--quiet')
        self.run([self.docker,'image','inspect','guanye-worker','--format','{{.Id}}'],'Verify existing local worker image')
        self.compose('up','-d','--no-build','otel','prometheus','grafana','mosquitto','device-bridge','worker','dispatcher')
        self.healthy('http://127.0.0.1:19090/-/ready');self.healthy('http://127.0.0.1:13000/api/health')
        self.run([sys.executable,'-u',str(ROOT/'scripts/smoke_extended.py'),'observability'],'Real monitoring and MQTT integration checks')
        logs=self.compose('logs','--tail','120','otel',capture=True)
        if not re.search(r'(?:"spans"\s*:\s*[1-9]|Spans\s*:\s*[1-9]|\bTraces\b)',logs):
            raise RuntimeError('Metrics passed but no trace export batch was found in collector logs')
        self.state['collector_trace_export']='pass: collector debug exporter received trace batches'

    def analytics(self):
        import psycopg,pymysql
        sys.path.insert(0,str(ROOT))
        from integrations.platform_bridge import read_config
        config=read_config(ROOT)
        with psycopg.connect(host='127.0.0.1',port=15432,dbname='guanye',user='guanye',password=config['POSTGRES_PASSWORD']) as db:
            active=db.execute("SELECT COUNT(*) FROM tasks WHERE status IN ('queued','running') AND kind NOT IN ('quality','dataset_audit')").fetchone()[0]
        if active:raise RuntimeError('Research tasks are active; finish them before switching to the analytics memory budget')
        self.compose('config','--quiet')
        self.compose('build','flink-jobmanager')
        # All three Flink services use the same build; tag the completed image to avoid rebuilding it.
        self.run([self.docker,'tag','guanye-flink-jobmanager','guanye-flink-taskmanager'],'Reuse Flink image for task manager')
        self.run([self.docker,'tag','guanye-flink-jobmanager','guanye-flink-submit'],'Reuse Flink image for SQL submission')
        self.compose('pull','kafka','doris')
        with psycopg.connect(host='127.0.0.1',port=15432,dbname='guanye',user='guanye',password=config['POSTGRES_PASSWORD']) as db:
            if db.execute("SELECT COUNT(*) FROM tasks WHERE status IN ('queued','running') AND kind NOT IN ('quality','dataset_audit')").fetchone()[0]:raise RuntimeError('Research tasks became active; wait for them before switching modes')
        self.compose('stop','models')
        self.state['research_model_runtime']='paused to keep the analytics stage within the VM memory budget'
        self.compose('up','-d','--no-build','kafka','doris','flink-jobmanager','flink-taskmanager')
        deadline=time.monotonic()+180
        while True:
            try:
                self.compose('exec','-T','kafka','/opt/kafka/bin/kafka-topics.sh','--bootstrap-server','kafka:9092','--list',capture=True)
                break
            except RuntimeError:
                if time.monotonic()>deadline:raise
                time.sleep(5)
        for topic in ('guanye.events','guanye.metrics.minute'):
            arguments=['exec','-T','kafka','/opt/kafka/bin/kafka-topics.sh','--bootstrap-server','kafka:9092','--create','--if-not-exists','--topic',topic,'--partitions','1','--replication-factor','1']
            if topic.endswith('.minute'):arguments+=['--config','cleanup.policy=compact']
            self.compose(*arguments,capture=True)
        self.healthy('http://127.0.0.1:18083/overview')
        deadline=time.monotonic()+180
        while True:
            try:
                db=pymysql.connect(host='127.0.0.1',port=19030,user='root',password=config.get('DORIS_PASSWORD',''),connect_timeout=3)
                with db.cursor() as cursor:cursor.execute('SELECT 1');cursor.fetchone()
                db.close();break
            except pymysql.Error:
                if time.monotonic()>deadline:raise
                time.sleep(5)
        self.compose('up','-d','--no-build','--force-recreate','otel','worker','dispatcher','analytics-sink')
        jobs=httpx.get('http://127.0.0.1:18083/jobs/overview',trust_env=False,timeout=10).json()['jobs']
        if not any(j['state']=='RUNNING' and j['name']=='guanye-minute-metrics' for j in jobs):
            self.compose('run','--rm','--no-deps','flink-submit')
        self.run([sys.executable,'-u',str(ROOT/'scripts/smoke_extended.py'),'analytics'],'Real Kafka / Flink SQL / Doris integration checks')

    def offline(self):
        import psycopg
        sys.path.insert(0,str(ROOT))
        from integrations.platform_bridge import read_config
        config=read_config(ROOT)
        arguments=dict(host='127.0.0.1',port=15432,dbname='guanye',user='guanye',password=config['POSTGRES_PASSWORD'])
        with psycopg.connect(**arguments,autocommit=True) as db:
            if not db.execute("SELECT 1 FROM pg_database WHERE datname='airflow'").fetchone():
                db.execute('CREATE DATABASE airflow')
        self.compose('config','--quiet')
        self.compose('build','airflow')
        with psycopg.connect(**arguments) as db:
            if db.execute("SELECT COUNT(*) FROM tasks WHERE status IN ('queued','running') AND kind NOT IN ('quality','dataset_audit')").fetchone()[0]:
                raise RuntimeError('Research tasks are active; finish them before switching to the offline memory budget')
        self.compose('stop','models','mlflow','analytics-sink','flink-taskmanager','flink-jobmanager','doris','kafka')
        self.state['memory_budget']='research and stream-analysis services paused; persistent volumes retained'
        self.compose('up','-d','--no-build','airflow')
        self.healthy('http://127.0.0.1:18084/api/v2/monitor/health',240)
        self.run([sys.executable,'-u',str(ROOT/'scripts/smoke_extended.py'),'offline'],'Real authenticated Airflow scheduler and S3 Parquet checks')

    def analytics_verify(self):
        self.compose('up','-d','--no-build','--force-recreate','analytics-sink')
        self.run([sys.executable,'-u',str(ROOT/'scripts/smoke_extended.py'),'analytics'],'Recheck Kafka / Flink SQL / Doris integration after source-only fix')

    def qwen(self):
        import psycopg
        sys.path.insert(0,str(ROOT))
        from integrations.platform_bridge import read_config
        from model_service.providers import weights_ready
        if not weights_ready(ROOT/'.runtime/models','qwen'):raise RuntimeError('Complete and verify the Qwen download first')
        config=read_config(ROOT)
        with psycopg.connect(host='127.0.0.1',port=15432,dbname='guanye',user='guanye',password=config['POSTGRES_PASSWORD']) as db:
            if db.execute("SELECT COUNT(*) FROM tasks WHERE status IN ('queued','running') AND kind NOT IN ('quality','dataset_audit')").fetchone()[0]:raise RuntimeError('A research task is active; finish it before switching modes')
        self.compose('stop','airflow','analytics-sink','flink-taskmanager','flink-jobmanager','doris','kafka','mlflow')
        self.run([self.docker,'image','inspect','guanye-models','--format','{{.Id}}'],'Verify existing public-model image')
        self.compose('up','-d','--no-build','--force-recreate','models')
        self.healthy('http://127.0.0.1:18082/health',120)
        self.run([sys.executable,'-u',str(ROOT/'scripts/smoke_extended.py'),'qwen'],'Actual queued Qwen3.5-4B evidence report and S3 readback')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['observability','analytics','analytics_verify','offline','qwen'],default='observability');args=parser.parse_args()
    runner=Runner(args.stage)
    try:
        getattr(runner,args.stage)();runner.state['status']='pass';runner.save('complete')
        print('STAGE PASSED. Detailed report: '+str(RUNTIME/(args.stage+'-verification.json')),flush=True)
        return 0
    except Exception as error:
        runner.state.update(status='failed',error=str(error));runner.save('failed')
        services={'analytics':['kafka','doris','flink-jobmanager','flink-taskmanager','analytics-sink'],'offline':['airflow'],'observability':['otel','prometheus','grafana','mosquitto','device-bridge','worker']}
        services['analytics_verify']=services['analytics']
        services['qwen']=['models','worker']
        try:runner.compose('logs','--tail','100',*services[args.stage],capture=True)
        except Exception:pass
        runner.save('failed');print('STAGE FAILED: '+str(error),flush=True)
        return 1
    finally:runner.log.close()


if __name__=='__main__':raise SystemExit(main())
