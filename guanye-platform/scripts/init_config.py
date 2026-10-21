"""Create project-local service secrets and a local development CA, never a user account."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
import argparse,base64,hashlib,hmac,ipaddress,json,secrets
from cryptography import x509
from cryptography.x509.oid import NameOID,ExtendedKeyUsageOID
from cryptography.hazmat.primitives import hashes,serialization
from cryptography.hazmat.primitives.asymmetric import rsa
root=Path(__file__).resolve().parents[1]
def jwt(secret,claims):
    def b(x):return base64.urlsafe_b64encode(json.dumps(x,separators=(',',':')).encode()).rstrip(b'=')
    first=b({'alg':'HS256','typ':'JWT'})+b'.'+b(claims);return (first+b'.'+base64.urlsafe_b64encode(hmac.new(secret.encode(),first,hashlib.sha256).digest()).rstrip(b'=')).decode()
def certificate(pki,name,ca=None,server=False):
    key=rsa.generate_private_key(public_exponent=65537,key_size=2048);now=datetime.now(timezone.utc);subject=x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,name)])
    issuer=subject if ca is None else ca[1].subject
    builder=x509.CertificateBuilder().subject_name(subject).issuer_name(issuer).public_key(key.public_key()).serial_number(x509.random_serial_number()).not_valid_before(now-timedelta(minutes=5)).not_valid_after(now+timedelta(days=365 if ca is None else 30))
    builder=builder.add_extension(x509.BasicConstraints(ca=ca is None,path_length=0 if ca is None else None),critical=True)
    if ca is not None:
        builder=builder.add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH if server else ExtendedKeyUsageOID.CLIENT_AUTH]),critical=False)
    if server:builder=builder.add_extension(x509.SubjectAlternativeName([x509.DNSName('localhost'),x509.DNSName('mosquitto'),x509.DNSName('guanye.local'),x509.IPAddress(ipaddress.ip_address('127.0.0.1'))]),critical=False)
    cert=builder.sign(key if ca is None else ca[0],hashes.SHA256())
    file='ca' if ca is None else 'mqtt' if server else 'bridge'
    (pki/(file+'.key')).write_bytes(key.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption()))
    (pki/(file+'.crt')).write_bytes(cert.public_bytes(serialization.Encoding.PEM));return key,cert
def main():
    runtime=root/'.runtime';runtime.mkdir(exist_ok=True);pki=runtime/'pki';pki.mkdir(exist_ok=True)
    path=root/'.env'
    if path.exists():
        existing=path.read_text(encoding='utf-8')
        for name in ['MODEL_SERVICE_SECRET','DEVICE_GATEWAY_SECRET']:
            if name+'=' not in existing:existing+='\n'+name+'='+secrets.token_urlsafe(36)+'\n'
        path.write_text(existing,encoding='utf-8')
        print('Existing service secrets preserved; missing internal service keys added.');return
    values={k:secrets.token_urlsafe(36) for k in ['POSTGRES_PASSWORD','REDIS_PASSWORD','S3_ACCESS_KEY','S3_SECRET_KEY','GUANYE_BRIDGE_SECRET','MODEL_SERVICE_SECRET','DEVICE_GATEWAY_SECRET','GRAFANA_PASSWORD','AIRFLOW_SECRET']}
    values['AIRFLOW_FERNET_KEY']=base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
    path.write_text('\n'.join(k+'='+v for k,v in values.items())+'\n',encoding='utf-8')
    (runtime/'s3.json').write_text(json.dumps({'identities':[{'name':'guanye','credentials':[{'accessKey':values['S3_ACCESS_KEY'],'secretKey':values['S3_SECRET_KEY']}],'actions':['Admin','Read','Write','List','Tagging']}]},indent=2))
    now=int(datetime.now(timezone.utc).timestamp());token=jwt(values['GUANYE_BRIDGE_SECRET'],{'sub':'prometheus','iss':'guanye-local','aud':['guanye-business'],'scope':'metrics','iat':now,'exp':now+365*86400})
    (runtime/'metrics-token').write_text(token)
    ca=certificate(pki,'Guanye local development CA');certificate(pki,'mosquitto',ca,True);certificate(pki,'platform-bridge',ca)
    for name in ['artifacts','models','uploads','downloads']:(runtime/name).mkdir(exist_ok=True)
    import shutil
    for directory,names in [('broker',['ca.crt','mqtt.crt','mqtt.key']),('client',['ca.crt','bridge.crt','bridge.key'])]:
        target=pki/directory;target.mkdir(exist_ok=True)
        for name in names:shutil.copy2(pki/name,target/name)
    print('Service configuration and local TLS certificates created. No Windows trust store was changed.')
if __name__=='__main__':main()
