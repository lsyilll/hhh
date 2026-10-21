from pathlib import Path
import urllib.request,hashlib,json,zipfile,concurrent.futures
root=Path(__file__).resolve().parents[1]/'.tooling';root.mkdir(exist_ok=True)
def get(url):return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'guanye-setup'}),timeout=180).read()
def download(name,url,expected=None,extract=False):
    try:
        path=root/name
        if not path.exists():path.write_bytes(get(url))
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if expected:assert digest==expected,(name,'checksum mismatch')
        if extract:
            folder=root/name.split('.')[0];folder.mkdir(exist_ok=True)
            with zipfile.ZipFile(path) as archive:archive.extractall(folder)
        print(name,'ready',flush=True);return {'name':name,'source':url,'sha256':digest,'verified_against_publisher':bool(expected),'status':'ready'}
    except Exception as e:print(name,str(e),flush=True);return {'name':name,'status':'failed','error':str(e)}
uv=json.loads(get('https://pypi.org/pypi/uv/json'));wheel=next(x for x in uv['urls'] if x['filename'].endswith('py3-none-win_amd64.whl'))
jobs=[('uv.whl',wheel['url'],wheel['digests']['sha256'],True),('nginx.zip','https://nginx.org/download/nginx-1.28.0.zip',None,True),('docker.zip','https://download.docker.com/win/static/stable/x86_64/docker-28.5.1.zip',None,True),('docker-compose.exe','https://github.com/docker/compose/releases/download/v2.39.4/docker-compose-windows-x86_64.exe',None,False)]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:result=list(pool.map(lambda args:download(*args),jobs))
(root/'more-tools-check.json').write_text(json.dumps(result,indent=2));print('Portable tool downloads recorded.')
