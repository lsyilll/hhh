"""Raspberry Pi / USB camera collector with MQTT 5 TLS and durable offline retry."""
import argparse,json,logging,os,sqlite3,ssl,time,uuid
from pathlib import Path
import requests,paho.mqtt.client as mqtt
def capture(path):
    try:
        from picamera2 import Picamera2
        camera=Picamera2();camera.configure(camera.create_still_configuration(main={'size':(1600,1200)}));camera.start();time.sleep(.7)
        try:camera.capture_file(str(path))
        finally:camera.stop();camera.close()
    except ImportError:
        import cv2
        camera=cv2.VideoCapture(0)
        try:
            ok,frame=camera.read()
            if not ok:raise RuntimeError('No camera frame')
            if not cv2.imwrite(str(path),frame):raise RuntimeError('Photo save failed')
        finally:camera.release()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--once',action='store_true');parser.add_argument('--interval',type=int,default=60);parser.add_argument('--gpio',action='store_true');args=parser.parse_args()
    config_path=Path(args.config).resolve();config=json.loads(config_path.read_text());folder=config_path.parent;spool=folder/'spool';spool.mkdir(exist_ok=True)
    db=sqlite3.connect(spool/'queue.sqlite3');db.execute('CREATE TABLE IF NOT EXISTS queue(id TEXT PRIMARY KEY,path TEXT NOT NULL,attempts INTEGER NOT NULL DEFAULT 0,next_try REAL NOT NULL DEFAULT 0)');db.commit()
    session=requests.Session();session.cert=(str(folder/'device.crt'),str(folder/'device.key'));session.verify=str(folder/'ca.crt')
    client=mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,client_id=config['device_id'],protocol=mqtt.MQTTv5);client.tls_set(str(folder/'ca.crt'),str(folder/'device.crt'),str(folder/'device.key'),tls_version=ssl.PROTOCOL_TLS_CLIENT)
    try:client.connect(config['mqtt_host'],int(config['mqtt_port']),60);client.loop_start()
    except OSError:logging.warning('Broker unavailable; local collection remains available')
    light=None;button=None
    if args.gpio:
        from gpiozero import Button,LED
        button=Button(17,pull_up=True,bounce_time=.15);light=LED(27)
    def shoot():
        key=uuid.uuid4().hex;path=spool/(key+'.jpg')
        try:
            if light:light.on()
            capture(path);db.execute('INSERT INTO queue(id,path) VALUES(?,?)',(key,str(path)));db.commit()
        finally:
            if light:light.off()
    def flush():
        for key,path,attempts in db.execute('SELECT id,path,attempts FROM queue WHERE next_try<=? ORDER BY rowid LIMIT 10',(time.time(),)).fetchall():
            try:
                with Path(path).open('rb') as photo:r=session.post(config['api_base']+'/device-api/upload',data={'sample_id':key},files={'image':('leaf.jpg',photo,'image/jpeg')},timeout=30)
                r.raise_for_status();db.execute('DELETE FROM queue WHERE id=?',(key,));db.commit();Path(path).unlink(missing_ok=True)
            except requests.RequestException:
                db.execute('UPDATE queue SET attempts=attempts+1,next_try=? WHERE id=?',(time.time()+min(3600,5*2**min(attempts,9)),key));db.commit();break
    if args.once:shoot();flush();client.loop_stop();return
    last_capture=0;last_heartbeat=0
    try:
        while True:
            now=time.time()
            if button:
                if button.is_pressed:shoot();button.wait_for_release()
            elif now-last_capture>=max(10,args.interval):shoot();last_capture=now
            flush()
            if now-last_heartbeat>=30:
                pending=db.execute('SELECT COUNT(*) FROM queue').fetchone()[0];client.publish('guanye/devices/'+config['device_id']+'/heartbeat',json.dumps({'time':int(now),'queued':pending}),qos=1);last_heartbeat=now
            time.sleep(.5)
    finally:client.loop_stop();client.disconnect();db.close()
if __name__=='__main__':main()
