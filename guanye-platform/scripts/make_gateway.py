from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'integrations'))
from platform_bridge import read_config
config=read_config(root);pki=(root/'.runtime/pki').as_posix();logs=(root/'.runtime').as_posix()
text='''worker_processes 1;
error_log LOGROOT/nginx-error.log;
pid LOGROOT/nginx.pid;
events { worker_connections 256; }
http {
  access_log off;
  client_max_body_size 21m;
  map $http_upgrade $connection_upgrade { default upgrade; '' close; }
  map $ssl_client_s_dn $device_id { default ''; ~CN=(?<device_cn>[0-9a-f-]+) $device_cn; }
  server {
    listen 127.0.0.1:8443 ssl;
    server_name localhost guanye.local;
    ssl_certificate PKI/mqtt.crt; ssl_certificate_key PKI/mqtt.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    location /device-api/ { return 404; }
    location / {
      proxy_pass http://127.0.0.1:7860;
      proxy_http_version 1.1; proxy_set_header Host $http_host;
      proxy_set_header X-Forwarded-Proto https;
      proxy_set_header Upgrade $http_upgrade; proxy_set_header Connection $connection_upgrade;
      proxy_buffering off; proxy_read_timeout 3600s;
      proxy_set_header X-Device-Gateway ''; proxy_set_header X-Client-Verify ''; proxy_set_header X-Device-Id '';
    }
  }
  server {
    listen 127.0.0.1:8444 ssl;
    server_name guanye.local localhost;
    ssl_certificate PKI/mqtt.crt; ssl_certificate_key PKI/mqtt.key;
    ssl_client_certificate PKI/ca.crt; ssl_verify_client on;
    ssl_protocols TLSv1.2 TLSv1.3;
    location /device-api/ {
      proxy_pass http://127.0.0.1:7860;
      proxy_set_header Host $http_host; proxy_set_header X-Forwarded-Proto https;
      proxy_set_header X-Client-Verify $ssl_client_verify; proxy_set_header X-Device-Id $device_id;
      proxy_set_header X-Device-Gateway BRIDGESECRET;
    }
    location / { return 404; }
  }
}
'''.replace('LOGROOT',logs).replace('PKI',pki).replace('BRIDGESECRET',config['DEVICE_GATEWAY_SECRET'])
(root/'.runtime/nginx.conf').write_text(text,encoding='utf-8');print('Private loopback HTTPS gateway config generated; no certificate trust settings changed.')
