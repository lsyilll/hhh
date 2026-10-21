"""Scoped service checks, using temporary QA projects created by smoke_core."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import ssl
import socket
import tempfile
import threading
import time
import uuid

import httpx
import psycopg
import pymysql

ROOT=Path(__file__).resolve().parents[1]


def eventually(function,seconds=100):
    deadline=time.monotonic()+seconds
    last=None
    while time.monotonic()<deadline:
        try:
            result=function()
            if result:return result
        except (httpx.HTTPError,AssertionError,KeyError,pymysql.Error) as error:last=error
        time.sleep(3)
    raise RuntimeError('Service check did not become ready: '+str(last or function.__name__))


def device_certificate(device,folder):
    from cryptography import x509
    from cryptography.x509.oid import NameOID,ExtendedKeyUsageOID
    from cryptography.hazmat.primitives import hashes,serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    pki=ROOT/'.runtime/pki'
    ca=x509.load_pem_x509_certificate((pki/'ca.crt').read_bytes())
    ca_key=serialization.load_pem_private_key((pki/'ca.key').read_bytes(),password=None)
    key=rsa.generate_private_key(public_exponent=65537,key_size=2048)
    now=datetime.now(timezone.utc)
    cert=(x509.CertificateBuilder().subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,device)]))
        .issuer_name(ca.subject).public_key(key.public_key()).serial_number(x509.random_serial_number())
        .not_valid_before(now-timedelta(minutes=1)).not_valid_after(now+timedelta(minutes=10))
        .add_extension(x509.BasicConstraints(ca=False,path_length=None),critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]),critical=False)
        .sign(ca_key,hashes.SHA256()))
    cert_path=folder/'device.crt';key_path=folder/'device.key'
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption()))
    return str(cert_path),str(key_path)


def mqtt_check(config,device,db_args):
    import paho.mqtt.client as mqtt
    runtime=ROOT/'.runtime/extended';runtime.mkdir(parents=True,exist_ok=True)
    ca=ROOT/'.runtime/pki/ca.crt'
    # TLS 1.2 makes missing-client-certificate rejection observable at handshake.
    context=ssl.create_default_context(cafile=str(ca));context.maximum_version=ssl.TLSVersion.TLSv1_2
    rejected=False
    try:
        with socket.create_connection(('127.0.0.1',18883),timeout=5) as raw:
            with context.wrap_socket(raw,server_hostname='localhost'):pass
    except ssl.SSLError:rejected=True
    assert rejected,'Broker accepted a TLS client without a certificate'
    with tempfile.TemporaryDirectory(prefix='mqtt-qa-',dir=runtime) as temporary:
        folder=Path(temporary)
        assert folder.resolve().is_relative_to(runtime.resolve())
        cert_path,key_path=device_certificate(device,folder)
        connected=threading.Event();published=threading.Event();reason=[]
        client=mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,client_id='qa-'+uuid.uuid4().hex,protocol=mqtt.MQTTv5)
        client.tls_set(str(ca),cert_path,key_path,tls_version=ssl.PROTOCOL_TLS_CLIENT)
        def on_connect(c,u,f,code,p):
            if not code.is_failure:connected.set()
        def on_publish(c,u,mid,code,p):reason.append(code);published.set()
        client.on_connect=on_connect;client.on_publish=on_publish
        with psycopg.connect(**db_args) as db:
            before=db.execute('SELECT last_seen FROM devices WHERE id=%s',(uuid.UUID(device),)).fetchone()['last_seen']
        client.connect('localhost',18883,30);client.loop_start()
        try:
            assert connected.wait(10),'Certificate-authenticated MQTT connection failed'
            client.publish('guanye/devices/'+device+'/heartbeat',json.dumps({'scope':'temporary software integration check'}),qos=1)
            assert published.wait(10) and not reason[-1].is_failure,'Own device heartbeat was rejected'
            def heartbeat_written():
                with psycopg.connect(**db_args) as db:
                    value=db.execute('SELECT last_seen FROM devices WHERE id=%s',(uuid.UUID(device),)).fetchone()['last_seen']
                return value is not None and (before is None or value>before)
            eventually(heartbeat_written,30)
            published.clear();client.publish('guanye/devices/'+str(uuid.uuid4())+'/heartbeat','{}',qos=1)
            assert published.wait(10) and reason[-1].is_failure,'MQTT topic ACL failed to reject another device identity'
        finally:client.disconnect();client.loop_stop()
    return {'mqtt5_tls_client_certificate':'pass','missing_client_certificate':'rejected','own_heartbeat_postgres_update':'pass','other_device_topic':'rejected','scope':'temporary software device; no physical camera or Raspberry Pi claimed'}


def observability(**context):
    config=context['config'];result={}
    with httpx.Client(trust_env=False,timeout=10) as client:
        def query(expression):
            response=client.get('http://127.0.0.1:19090/api/v1/query',params={'query':expression})
            response.raise_for_status();data=response.json();assert data['status']=='success'
            return data['data']['result']
        eventually(lambda:any(float(row['value'][1])==1 for row in query('up{job="business"}')),100)
        result['prometheus_spring_authenticated_scrape']='pass'
        eventually(lambda:any(float(row['value'][1])>0 for row in query('guanye_tasks_total')),100)
        result['opentelemetry_worker_metric_to_prometheus']='pass: actual quality/audit task execution counter'
        auth=httpx.BasicAuth('admin',config['GRAFANA_PASSWORD'])
        response=client.get('http://127.0.0.1:13000/api/dashboards/uid/guanye-runtime',auth=auth)
        response.raise_for_status();dashboard=response.json()['dashboard'];assert len(dashboard['panels'])>=4
        response=client.get('http://127.0.0.1:13000/api/datasources',auth=auth);response.raise_for_status()
        datasource=next(row for row in response.json() if row['type']=='prometheus')
        response=client.get('http://127.0.0.1:13000/api/datasources/proxy/uid/'+datasource['uid']+'/api/v1/query',params={'query':'up{job="business"}'},auth=auth)
        response.raise_for_status();assert response.json()['data']['result']
        result['grafana_provisioned_dashboard_and_datasource']='pass: 4 panels and live Prometheus query'
    result['devices']=mqtt_check(config,context['device'],context['db_args'])
    return result


def check(stage,**context):
    if stage=='observability':return observability(**context)
    if stage=='analytics':return analytics(**context)
    if stage=='data_versioning':
        from smoke_data_versioning import check as data_check
        return data_check(**context)
    if stage=='offline':
        from smoke_airflow import check as airflow_check
        return airflow_check(**context)
    if stage=='qwen':
        from smoke_qwen import check as qwen_check
        return qwen_check(**context)
    if stage=='gateway':
        from smoke_gateway import check as gateway_check
        return gateway_check(**context)
    raise ValueError('Unknown extended validation stage')


def analytics(**context):
    import pymysql
    project=context['project'];db_args=context['db_args'];result={}
    with httpx.Client(trust_env=False,timeout=10) as client:
        def running():
            response=client.get('http://127.0.0.1:18083/jobs/overview');response.raise_for_status()
            return any(j['state']=='RUNNING' and j['name']=='guanye-minute-metrics' for j in response.json()['jobs'])
        eventually(running,100)
    doris=pymysql.connect(host='127.0.0.1',port=19030,user='root',password=context['config'].get('DORIS_PASSWORD',''),autocommit=True,connect_timeout=5,read_timeout=15,cursorclass=pymysql.cursors.DictCursor)
    def rows(sql,params=()):
        with doris.cursor() as cursor:cursor.execute(sql,params);return cursor.fetchall()
    try:
        def events_delivered():
            with psycopg.connect(**db_args) as db:
                events=db.execute('SELECT id,event_type,created,kafka_sent FROM outbox WHERE project_id=%s ORDER BY created',(uuid.UUID(project),)).fetchall()
            return events if events and all(e['kafka_sent'] for e in events) else None
        events=eventually(events_delivered,120)
        complete=[e for e in events if e['event_type']=='task.complete'];assert complete
        first=complete[0];event_id=str(first['id'])
        eventually(lambda:rows('SELECT event_id FROM guanye.ods_events WHERE event_id=%s',(event_id,)),120)
        result['spring_outbox_kafka_to_doris_ods']='pass'
        # Re-send exactly one QA event without dispatching its task again.
        with psycopg.connect(**db_args) as db:
            db.execute('UPDATE outbox SET kafka_sent=false WHERE id=%s AND project_id=%s',(first['id'],uuid.UUID(project)))
        eventually(events_delivered,60)
        assert rows('SELECT COUNT(*) AS n FROM guanye.ods_events WHERE event_id=%s',(event_id,))[0]['n']==1
        result['duplicate_event_id_doris_upsert']='pass: one row after re-delivery'
        window_epoch=int(first['created'].timestamp())//60*60
        from zoneinfo import ZoneInfo
        expected_start=datetime.fromtimestamp(window_epoch,timezone.utc).astimezone(ZoneInfo('Asia/Shanghai')).replace(tzinfo=None)
        expected=sum(1 for e in complete if int(e['created'].timestamp())//60*60==window_epoch)
        # A later real task advances the Kafka source watermark past the first window.
        delay=max(0,window_epoch+73-time.time())
        if delay:time.sleep(delay)
        context['task']('quality',context['sample'])
        def window_ready():
            found=rows('SELECT completed,failed,window_start FROM guanye.dws_project_minute WHERE project_id=%s AND window_start=%s',(project,expected_start))
            return found[0] if found and int(found[0]['completed'])==expected else None
        aggregate=eventually(window_ready,150)
        result['flink_sql_dedup_and_minute_window_to_doris_dws']={'status':'pass','completed':int(aggregate['completed']),'expected_unique_completed':expected,'window_start':str(aggregate['window_start'])}
        assert rows('SELECT COUNT(*) AS n FROM guanye.dwd_tasks WHERE project_id=%s',(project,))[0]['n']>=expected
        daily=rows('SELECT tasks,failures FROM guanye.ads_daily_quality WHERE project_id=%s',(project,));assert daily and int(daily[0]['tasks'])>=expected
        result['doris_dwd_and_ads_views']='pass: real task events and daily aggregation'
        result['kafka_qa_event_retention']='Temporary QA events remain in the shared topic until normal Kafka retention; no real events were deleted'
        return result
    finally:
        doris.close()


def cleanup(stage,project,config):
    if stage!='analytics':return
    db=pymysql.connect(host='127.0.0.1',port=19030,user='root',password=config.get('DORIS_PASSWORD',''),autocommit=True,connect_timeout=5,read_timeout=15)
    try:
        with db.cursor() as cursor:
            for table in ('dws_project_minute','ods_events'):
                cursor.execute('DELETE FROM guanye.'+table+' WHERE project_id=%s',(project,))
    finally:db.close()


if __name__=='__main__':
    import argparse,os
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['observability','analytics','data_versioning','offline','qwen','gateway']);args=parser.parse_args()
    os.environ['GUANYE_CHECK_EXTENDED']=args.stage
    import smoke_core
    raise SystemExit(smoke_core.main())
