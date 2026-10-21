"""Nginx TLS and mTLS roundtrip; trust only this request's local CA."""
from pathlib import Path
import socket
import ssl
import tempfile
import time
import httpx
import psycopg
import uuid

ROOT=Path(__file__).resolve().parents[1]

def check(**context):
    from smoke_extended import device_certificate
    ca=ROOT/'.runtime/pki/ca.crt';tls=ssl.create_default_context(cafile=str(ca))
    with httpx.Client(verify=tls,trust_env=False,timeout=10) as client:
        response=client.get('https://localhost:8443/');response.raise_for_status()
        assert response.text=='Temporary Guanye TLS integration adapter'
        assert client.post('https://localhost:8443/device-api/heartbeat',headers={'X-Client-Verify':'SUCCESS','X-Device-Id':context['device']}).status_code==404
        response=client.post('https://localhost:8444/device-api/heartbeat')
        assert response.status_code==400
    assert httpx.post('http://127.0.0.1:7860/device-api/heartbeat',headers={'X-Client-Verify':'SUCCESS','X-Device-Gateway':'forged','X-Device-Id':context['device']},trust_env=False).status_code==401
    runtime=ROOT/'.runtime/extended'
    with tempfile.TemporaryDirectory(prefix='mtls-qa-',dir=runtime) as temporary:
        folder=Path(temporary);assert folder.resolve().is_relative_to(runtime.resolve())
        cert,key=device_certificate(context['device'],folder)
        tls=ssl.create_default_context(cafile=str(ca));tls.load_cert_chain(cert,key)
        with httpx.Client(verify=tls,trust_env=False,timeout=15) as client:
            response=client.post('https://localhost:8444/device-api/heartbeat',json={});response.raise_for_status()
        with psycopg.connect(**context['db_args']) as db:
            row=db.execute('SELECT last_seen FROM devices WHERE id=%s',(uuid.UUID(context['device']),)).fetchone()
            assert row['last_seen'] is not None
    return {'status':'pass','nginx_https_proxy':'pass','certificate_validation':'private CA explicitly trusted by test request; no system trust change','mtls_missing_client_certificate':'rejected with HTTP 400','public_port_device_route':'404','forged_direct_gateway_header':'401','approved_certificate_to_spring_postgres_heartbeat':'pass','scope':'temporary loopback adapter using actual device routes; no physical device or face/camera claim'}
