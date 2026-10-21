"""Transactional outbox: broker failures cannot lose database events or task requests."""
import json,logging,os,time
from worker.common import connection
from worker.jobs import app
def cycle():
    try:
        from kafka import KafkaProducer
        producer=KafkaProducer(bootstrap_servers=os.getenv('KAFKA_BOOTSTRAP','kafka:9092'),max_block_ms=1500,request_timeout_ms=1500,api_version_auto_timeout_ms=1500,value_serializer=lambda x:json.dumps(x,ensure_ascii=False).encode())
    except Exception:producer=None
    try:
        with connection() as db:
            rows=db.execute('SELECT * FROM outbox WHERE dispatched=false OR kafka_sent=false ORDER BY created LIMIT 50 FOR UPDATE SKIP LOCKED').fetchall()
            for row in rows:
                if not row['dispatched']:
                    if row['event_type']=='task.created':app.send_task('guanye.run',args=[row['payload']['task_id']],task_id=row['payload']['task_id'])
                    db.execute('UPDATE outbox SET dispatched=true WHERE id=%s',(row['id'],))
                if producer and not row['kafka_sent']:
                    event={'event_id':str(row['id']),'event_type':row['event_type'],'owner_id':row['owner_id'],'project_id':str(row['project_id']),
                           'ts_ms':int(row['created'].timestamp()*1000),**row['payload']}
                    producer.send('guanye.events',key=str(row['project_id']).encode(),value=event).get(timeout=3)
                    db.execute('UPDATE outbox SET kafka_sent=true WHERE id=%s',(row['id'],))
    finally:
        if producer:producer.close(timeout=2)
if __name__=='__main__':
    while True:
        try:cycle()
        except Exception:logging.exception('Outbox will retry')
        time.sleep(3)
