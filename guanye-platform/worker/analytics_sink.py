import json,os,time,logging,base64,socket,hashlib
from pathlib import Path
import pymysql,httpx
def initialize():
    db=pymysql.connect(host=os.getenv('DORIS_HOST','doris'),port=9030,user='root',password=os.getenv('DORIS_PASSWORD',''),autocommit=True)
    with db.cursor() as cursor:
        for sql in Path('/sql/schema.sql').read_text().split(';'):
            if sql.strip():cursor.execute(sql)
    db.close()
def stream_load_target(location, original, host):
    from urllib.parse import urlsplit,urlunsplit
    parsed=urlsplit(location)
    allowed={host,'127.0.0.1','localhost',*[item[4][0] for item in socket.getaddrinfo(host,None)]}
    if parsed.scheme!='http' or parsed.hostname not in allowed or parsed.path!=urlsplit(original).path or parsed.port!=8040 or parsed.fragment:
        raise ValueError('Unexpected Doris redirect target')
    # Doris may include Basic credentials in Location. Discard them and use our
    # configured Authorization header only after validating the internal target.
    return urlunsplit(('http',host+':8040',parsed.path,parsed.query,''))

def stream_load_label(topic,partition,offset):
    identity=f'{topic}:{partition}:{offset}'
    return 'guanye-'+hashlib.sha256(identity.encode()).hexdigest()

def load(table,data,label):
    # Doris FE redirects to BE; preserve authorization for that fixed internal service redirect only.
    auth='Basic '+base64.b64encode(('root:'+os.getenv('DORIS_PASSWORD','')).encode()).decode()
    url=f'http://{os.getenv("DORIS_HOST","doris")}:8030/api/guanye/{table}/_stream_load'
    headers={'Authorization':auth,'label':label,'format':'json','strip_outer_array':'true','Expect':'100-continue'}
    with httpx.Client(timeout=30,trust_env=False) as client:
        r=client.put(url,content=json.dumps([data]),headers=headers)
        if r.status_code==307:
            redirect=stream_load_target(r.headers['location'],url,os.getenv('DORIS_HOST','doris'))
            r=client.put(redirect,content=json.dumps([data]),headers=headers)
        r.raise_for_status();result=r.json()
        if result.get('Status') not in ('Success','Publish Timeout','Label Already Exists'):raise RuntimeError(result.get('Message','Doris load failed'))
if __name__=='__main__':
    from kafka import KafkaConsumer
    while True:
        try:initialize();break
        except Exception:logging.exception('Waiting for Doris');time.sleep(10)
    consumer=KafkaConsumer('guanye.events','guanye.metrics.minute',bootstrap_servers=os.environ['KAFKA_BOOTSTRAP'],group_id='guanye-doris',enable_auto_commit=False,auto_offset_reset='earliest')
    for message in consumer:
        if message.value is None:
            consumer.commit();continue
        data=json.loads(message.value)
        # Ignore late technical-QA windows after their isolated project has been cleaned.
        if str(data.get('owner_id','')).startswith('qa-'):
            from worker.common import connection
            with connection() as db:exists=db.execute('SELECT 1 FROM projects WHERE id=%s',(data['project_id'],)).fetchone()
            if not exists:
                consumer.commit();continue
        if message.topic=='guanye.events':
            row={k:data.get(k) for k in ['event_id','event_type','project_id','owner_id','task_id','status','duration_ms','ts_ms']};row['payload']=json.dumps(data,ensure_ascii=False);table='ods_events'
        else:row=data;table='dws_project_minute'
        load(table,row,stream_load_label(message.topic,message.partition,message.offset));consumer.commit()
