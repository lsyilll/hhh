import base64,hashlib,hmac,json,time,types
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'integrations'))
import pytest
from fastapi import FastAPI,HTTPException,Request
from fastapi.testclient import TestClient
from platform_bridge import token,read_config,install_platform_routes
def test_bridge_token_short_lived_and_scoped():
    key='a'*64;value=token({'id':'owner','name':'测试'},key);head,body,sig=value.split('.')
    expected=base64.urlsafe_b64encode(hmac.new(key.encode(),(head+'.'+body).encode(),hashlib.sha256).digest()).rstrip(b'=').decode();assert hmac.compare_digest(sig,expected)
    payload=json.loads(base64.urlsafe_b64decode(body+'='*(-len(body)%4)));assert payload['sub']=='owner' and payload['scope']=='business' and payload['aud']==['guanye-business']
    assert 0<payload['exp']-time.time()<=120
    device=json.loads(base64.urlsafe_b64decode(token({'id':'dev','name':'device'},key,'device').split('.')[1]+'=='));assert device['scope']=='device'
def test_configuration_does_not_evaluate_shell(tmp_path):
    (tmp_path/'.env').write_text('SECRET=$(do-not-execute)\n# ignore\nENDPOINT=http://127.0.0.1\n');assert read_config(tmp_path)['SECRET']=='$(do-not-execute)'
@pytest.fixture
def client(tmp_path):
    platform=tmp_path/'platform';(platform/'web/dist').mkdir(parents=True);(platform/'web/dist/index.html').write_text('platform fixture');(platform/'stack.catalog.json').write_text('{"technologies":[]}')
    app=FastAPI()
    def require(request):
        if request.headers.get('x-test-fixture')!='allowed':raise HTTPException(401,'test gate')
        return {'id':'owner','name':'test','admin':True}
    async def body(request):return await request.json()
    install_platform_routes(app,types.SimpleNamespace(ROOT=tmp_path),require,body)
    with TestClient(app) as client:yield client
def test_platform_data_requires_session(client):
    for path in ['/platform/','/api/platform/catalog','/api/platform/health','/api/platform/business/projects']:
        assert client.get(path).status_code==401
def test_uninitialized_platform_is_explicit(client):
    assert client.get('/platform/',headers={'x-test-fixture':'allowed'}).status_code==200
    result=client.get('/api/platform/business/projects',headers={'x-test-fixture':'allowed'});assert result.status_code==503 and '初始化' in result.json()['detail']
def test_device_api_cannot_skip_tls_gateway(client):
    # No initialized secret means no device upload endpoint can be used as a passwordless bypass.
    assert client.post('/device-api/heartbeat').status_code==503
