import json,os,ssl,uuid
import paho.mqtt.client as mqtt
from worker.common import connection
def message(client,userdata,packet):
    try:
        pieces=packet.topic.split('/')
        if len(pieces)!=4 or pieces[:2]!=['guanye','devices'] or pieces[3]!='heartbeat' or len(packet.payload)>64000:return
        key=uuid.UUID(pieces[2]);data=json.loads(packet.payload)
        with connection() as db:
            device=db.execute("SELECT * FROM devices WHERE id=%s AND status='approved'",(key,)).fetchone()
            if not device:return
            db.execute('UPDATE devices SET last_seen=now() WHERE id=%s',(key,))
            # Device-controlled text is not forwarded into privileged commands or model prompts.
    except (ValueError,KeyError,TypeError):return
client=mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,client_id='guanye-platform-bridge',protocol=mqtt.MQTTv5)
client.tls_set('/certs/ca.crt','/certs/bridge.crt','/certs/bridge.key',tls_version=ssl.PROTOCOL_TLS_CLIENT)
client.on_connect=lambda c,u,f,r,p:c.subscribe('guanye/devices/+/heartbeat',qos=1)
client.on_message=message
if __name__=='__main__':client.connect(os.getenv('MQTT_HOST','mosquitto'),int(os.getenv('MQTT_PORT','8883')),60);client.loop_forever()
