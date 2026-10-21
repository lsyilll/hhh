"""Authenticated same-origin adapter between the retained AgriChat UI and the platform."""
import asyncio,base64,hashlib,hmac,io,json,os,socket,time,uuid,zipfile
from pathlib import Path
from functools import lru_cache
import httpx
from fastapi import HTTPException,Request
from fastapi.responses import FileResponse,Response,StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

def read_config(folder):
    result={}
    if (folder/'.env').exists():
        for line in (folder/'.env').read_text(encoding='utf-8').splitlines():
            if '=' in line and not line.lstrip().startswith('#'):key,value=line.split('=',1);result[key]=value
    return result
def token(user,secret,scope='business'):
    def enc(value):return base64.urlsafe_b64encode(json.dumps(value,separators=(',',':')).encode()).rstrip(b'=')
    now=int(time.time());head=enc({'alg':'HS256','typ':'JWT'})+b'.'+enc({'sub':user['id'],'name':user['name'],'iss':'guanye-local','aud':['guanye-business'],'scope':scope,'iat':now,'exp':now+120})
    return (head+b'.'+base64.urlsafe_b64encode(hmac.new(secret.encode(),head,hashlib.sha256).digest()).rstrip(b'=')).decode()
def install_platform_routes(app,backend,require,json_body):
    root=Path(backend.ROOT)/'platform';health_cache={}
    def cfg():
        result=read_config(root)
        if len(result.get('GUANYE_BRIDGE_SECRET',''))<43:raise HTTPException(503,'完整平台尚未初始化，请先运行 platform/scripts/init_config.py。原图像问答可继续使用。')
        return result
    async def business(path,user,method='GET',data=None):
        config=cfg()
        if any(value in path for value in ('..','://','\\')):raise HTTPException(400,'不支持的接口路径。')
        try:
            async with httpx.AsyncClient(base_url='http://127.0.0.1:18081/v1/',timeout=15,trust_env=False) as client:
                result=await client.request(method,path,content=data,headers={'Authorization':'Bearer '+token(user,config['GUANYE_BRIDGE_SECRET']),'Content-Type':'application/json'})
        except httpx.TransportError:raise HTTPException(503,'Spring 业务服务尚未启动或无法连接。请启动完整平台基础服务。') from None
        if result.status_code>=400:
            try:detail=result.json().get('message','操作未完成。')
            except ValueError:detail='业务服务返回错误。'
            raise HTTPException(result.status_code,detail)
        return result.json()
    def storage():
        import boto3
        from botocore.config import Config
        config=cfg();return boto3.client('s3',endpoint_url='http://127.0.0.1:18333',aws_access_key_id=config['S3_ACCESS_KEY'],aws_secret_access_key=config['S3_SECRET_KEY'],config=Config(signature_version='s3v4',s3={'addressing_style':'path'},connect_timeout=3,read_timeout=20,retries={'max_attempts':1}))
    @app.get('/platform')
    @app.get('/platform/')
    async def page(request:Request):
        require(request);path=root/'web/dist/index.html'
        if not path.exists():raise HTTPException(503,'平台前端尚未构建。请运行 platform/scripts/build_web.ps1。')
        return FileResponse(path)
    if (root/'web/dist/assets').exists():app.mount('/platform/assets',StaticFiles(directory=root/'web/dist/assets'),name='platform-assets')
    @app.get('/api/platform/catalog')
    async def catalog(request:Request):
        require(request);return json.loads((root/'stack.catalog.json').read_text(encoding='utf-8'))
    @app.get('/api/platform/health')
    async def health(request:Request):
        require(request)
        if time.monotonic()-health_cache.get('time',0)<15:return health_cache['value']
        def probe():
            specs=[('business','Spring 业务服务',18081),('postgres','PostgreSQL',15432),('redis','Redis / Celery',16379),('s3','SeaweedFS / S3',18333),('mqtt','MQTT / Mosquitto',18883),('flink','Flink',18083),('doris','Doris',19030),('mlflow','MLflow',15000),('airflow','Airflow',18084),('grafana','Grafana',13000),('prometheus','Prometheus',19090),('models','独立模型服务',18082)]
            result=[]
            for key,name,port in specs:
                try:
                    with socket.create_connection(('127.0.0.1',port),timeout=.15):state='reachable'
                    if key in ('business','models'):
                        r=httpx.get('http://127.0.0.1:'+str(port)+('/actuator/health' if key=='business' else '/health'),timeout=1,trust_env=False)
                        if r.is_success:state='ready'
                except (OSError,httpx.HTTPError):state='stopped'
                result.append({'id':key,'name':name,'state':state})
            import shutil
            docker=shutil.which('docker') or next((str(p) for p in [Path(os.environ.get('USERPROFILE','C:/Users/12203'))/'AppData/Local/Programs/DockerDesktop/resources/bin/docker.exe',Path('C:/Program Files/Docker/Docker/resources/bin/docker.exe')] if p.is_file()),None)
            return {'services':result,'docker_available':bool(docker),'initialized':(root/'.env').exists(),'checked_at':time.time()}
        value=await run_in_threadpool(probe);health_cache.update(time=time.monotonic(),value=value);return value
    @app.api_route('/api/platform/business/{path:path}',methods=['GET','POST','PATCH'])
    async def proxy(path:str,request:Request):
        user=require(request);raw=await request.body()
        if len(raw)>150000:raise HTTPException(413,'请求体过大。')
        suffix=path+('?' + request.url.query if request.url.query else '')
        return await business(suffix,user,request.method,raw or None)
    @app.post('/api/platform/import-sample/{key}')
    async def publish(key:str,request:Request):
        user=require(request);data=await json_body(request)
        try:project=str(uuid.UUID(data.get('project_id','')))
        except ValueError:raise HTTPException(400,'请选择项目。') from None
        # Check project role before uploading bytes to object storage.
        projects=await business('projects',user)
        if not any(p['id']==project and p['role'] in ('ADMIN','COLLECTOR') for p in projects):raise HTTPException(403,'没有向该项目添加样本的权限。')
        item=app.state.samples.get(key,user)
        if item['user_id']!=user['id']:raise HTTPException(403,'请由样本所有者明确加入项目。')
        raw=Path(item['image']).read_bytes();sha=hashlib.sha256(raw).hexdigest();object_key='users/'+user['id']+'/samples/'+sha+Path(item['image']).suffix
        def upload():
            from botocore.exceptions import ClientError
            client=storage()
            try:client.head_bucket(Bucket='guanye')
            except ClientError as error:
                if str(error.response['Error']['Code']) not in ('404','NoSuchBucket'):raise
                client.create_bucket(Bucket='guanye')
            client.put_object(Bucket='guanye',Key=object_key,Body=raw,ContentType='image/'+('png' if object_key.endswith('.png') else 'webp' if object_key.endswith('.webp') else 'jpeg'))
        try:await run_in_threadpool(upload)
        except Exception:raise HTTPException(503,'SeaweedFS 对象存储尚未连接，样本未发布。') from None
        conversation=app.state.store.conversation(item['conversation_id'],user['id']);answer='\n\n'.join(m['text'] for m in conversation['messages'] if m['role']=='assistant')
        return await business('samples',user,'POST',json.dumps({'project_id':project,'source_id':key,'object_key':object_key,'sha256':sha,'quality':item['quality'],'model_answer':answer}).encode())
    @app.get('/api/platform/object')
    async def object_image(sample_id:uuid.UUID,request:Request):
        user=require(request);item=await business('samples/'+str(sample_id),user)
        try:
            result=await run_in_threadpool(lambda:storage().get_object(Bucket='guanye',Key=item['object_key']));raw=await run_in_threadpool(result['Body'].read,20*1024*1024+1)
        except Exception:raise HTTPException(503,'对象存储暂不可用。') from None
        if len(raw)>20*1024*1024:raise HTTPException(413,'图片超过大小限制。')
        return Response(raw,media_type=result.get('ContentType','application/octet-stream'),headers={'Cache-Control':'no-store'})
    @app.get('/api/platform/events')
    async def events(request:Request):
        user=require(request)
        async def stream():
            previous=None
            while not await request.is_disconnected():
                try:
                    rows=await business('tasks',user);value=json.dumps(rows,ensure_ascii=False,default=str)
                    if value!=previous:yield 'event: tasks\ndata: '+value+'\n\n';previous=value
                except HTTPException as error:yield 'event: service\ndata: '+json.dumps({'available':False,'message':error.detail},ensure_ascii=False)+'\n\n'
                yield ': heartbeat\n\n';await asyncio.sleep(5)
        return StreamingResponse(stream(),media_type='text/event-stream',headers={'Cache-Control':'no-store','X-Accel-Buffering':'no'})
    @app.get('/api/platform/task-artifact')
    async def task_artifact(task_id:uuid.UUID,key:str,request:Request):
        user=require(request);task=await business('tasks/'+str(task_id),user)
        def strings(value):
            if isinstance(value,str):return [value]
            if isinstance(value,dict):return [s for v in value.values() for s in strings(v)]
            if isinstance(value,list):return [s for v in value for s in strings(v)]
            return []
        prefix='users/'+task['owner_id']+'/tasks/'+str(task_id)+'/'
        if not key.startswith(prefix) or key not in strings(task.get('result')):raise HTTPException(404,'没有找到可访问的任务文件。')
        try:
            result=await run_in_threadpool(lambda:storage().get_object(Bucket='guanye',Key=key));raw=await run_in_threadpool(result['Body'].read,100*1024*1024+1)
        except Exception:raise HTTPException(503,'任务文件存储暂不可用。') from None
        if len(raw)>100*1024*1024:raise HTTPException(413,'文件超过下载限制。')
        name=Path(key).name.replace('"','').replace('\r','').replace('\n','')
        return Response(raw,media_type='application/octet-stream',headers={'Content-Disposition':'attachment; filename="'+name+'"','Cache-Control':'no-store'})
    @app.get('/api/platform/device-certificate/{key}')
    async def device_certificate(key:uuid.UUID,request:Request):
        user=require(request);item=await business('devices/'+str(key),user);projects=await business('projects',user)
        if item['status']!='approved' or not any(p['id']==item['project_id'] and p['role']=='ADMIN' for p in projects):raise HTTPException(403,'只有项目管理员能为已批准设备领取证书。')
        def issue():
            from cryptography import x509
            from cryptography.x509.oid import NameOID,ExtendedKeyUsageOID
            from cryptography.hazmat.primitives import hashes,serialization
            from cryptography.hazmat.primitives.asymmetric import rsa
            from datetime import datetime,timedelta,timezone
            pki=root/'.runtime/pki';ca=x509.load_pem_x509_certificate((pki/'ca.crt').read_bytes());ca_key=serialization.load_pem_private_key((pki/'ca.key').read_bytes(),None);private=rsa.generate_private_key(public_exponent=65537,key_size=2048);now=datetime.now(timezone.utc)
            cert=x509.CertificateBuilder().subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME,str(key))])).issuer_name(ca.subject).public_key(private.public_key()).serial_number(x509.random_serial_number()).not_valid_before(now-timedelta(minutes=1)).not_valid_after(now+timedelta(hours=24)).add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]),critical=False).sign(ca_key,hashes.SHA256())
            output=io.BytesIO()
            with zipfile.ZipFile(output,'w') as archive:
                archive.writestr('ca.crt',(pki/'ca.crt').read_bytes());archive.writestr('device.crt',cert.public_bytes(serialization.Encoding.PEM));archive.writestr('device.key',private.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption()));archive.writestr('device.json',json.dumps({'device_id':str(key),'project_id':item['project_id'],'mqtt_host':'guanye.local','mqtt_port':18883,'api_base':'https://guanye.local:8444','expires':(now+timedelta(hours=24)).isoformat()}))
            return output.getvalue()
        return Response(await run_in_threadpool(issue),media_type='application/zip',headers={'Content-Disposition':'attachment; filename="guanye-device-credentials.zip"','Cache-Control':'no-store'})
    async def device_call(request,path,method='GET',data=None):
        config=cfg()
        if request.headers.get('x-client-verify')!='SUCCESS' or not hmac.compare_digest(request.headers.get('x-device-gateway',''),config['DEVICE_GATEWAY_SECRET']):raise HTTPException(401,'设备必须通过 TLS 客户端证书入口连接。')
        try:key=str(uuid.UUID(request.headers.get('x-device-id','')))
        except ValueError:raise HTTPException(401,'设备证书编号不正确。') from None
        try:
            async with httpx.AsyncClient(timeout=15,trust_env=False) as client:
                r=await client.request(method,'http://127.0.0.1:18081/device/'+path,json=data,headers={'Authorization':'Bearer '+token({'id':key,'name':'device'},config['GUANYE_BRIDGE_SECRET'],'device')})
            if not r.is_success:raise HTTPException(r.status_code,'设备未获批准、已暂停或请求不正确。')
            return r.json()
        except httpx.TransportError:raise HTTPException(503,'设备业务服务暂不可用。') from None
    @app.post('/device-api/heartbeat')
    async def device_heartbeat(request:Request):return await device_call(request,'heartbeat','POST',{})
    @app.post('/device-api/upload')
    async def device_upload(request:Request):
        item=await device_call(request,'session');form=await request.form();image=form.get('image');sample_id=form.get('sample_id')
        if not image or not hasattr(image,'read'):raise HTTPException(400,'需要植物图片。')
        try:sample_id=uuid.UUID(str(sample_id)).hex
        except ValueError:raise HTTPException(400,'采集编号无效。') from None
        raw=await image.read(20*1024*1024+1);await image.close()
        if len(raw)>20*1024*1024:raise HTTPException(413,'图片超过20MB。')
        from PIL import Image,ImageOps
        try:
            with Image.open(io.BytesIO(raw)) as source:
                if source.width*source.height>40_000_000 or source.format not in ('JPEG','PNG','WEBP'):raise ValueError()
                converted=ImageOps.exif_transpose(source).convert('RGB');converted.thumbnail((1600,1600));buffer=io.BytesIO();converted.save(buffer,format='JPEG',quality=92);raw=buffer.getvalue()
        except Exception:raise HTTPException(400,'图片无法解码。') from None
        sha=hashlib.sha256(raw).hexdigest();key='users/'+item['owner_id']+'/devices/'+str(item['id'])+'/'+sha+'.jpg'
        try:await run_in_threadpool(lambda:storage().put_object(Bucket='guanye',Key=key,Body=raw,ContentType='image/jpeg'))
        except Exception:raise HTTPException(503,'图片存储暂不可用，请在设备本地保留并重试。') from None
        return await device_call(request,'sample','POST',{'project_id':item['project_id'],'source_id':uuid.UUID(str(item['id'])).hex+sample_id,'object_key':key,'sha256':sha,'quality':{},'model_answer':''})
