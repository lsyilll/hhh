"""Verified official Windows SDK wheels, installed outside the application environment."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request
import urllib.parse
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename
from network_env import windows_download_env

ROOT=Path(__file__).resolve().parents[1]

def main():
    env,_=windows_download_env();opener=urllib.request.build_opener(urllib.request.ProxyHandler({'https':env['HTTPS_PROXY']}))
    folder=ROOT/'.tooling/tensorrt-wheels';folder.mkdir(parents=True,exist_ok=True)
    records=[]
    for package,marker in [('tensorrt-cu12-bindings','cp311-none-win_amd64'),('tensorrt-cu12-libs','py2.py3-none-win_amd64')]:
        base='https://pypi.nvidia.com/'+package+'/'
        with opener.open(base,timeout=45) as response:html=response.read().decode()
        href=next(h for h in re.findall(r'href=[\"\x27]([^\"\x27]+)',html) if '10.8.0.43-' in h and marker in h)
        url=urllib.parse.urljoin(base,href);parts=urllib.parse.urlsplit(url)
        records.append({'url':urllib.parse.urlunsplit(parts._replace(fragment='')),'sha256':parts.fragment.split('=',1)[1],'filename':parts.path.rsplit('/',1)[1]})
    order={tag:i for i,tag in enumerate(sys_tags())}
    for package,version in [('cuda-bindings','12.8.0'),('nvidia-cuda-runtime-cu12','12.8.90')]:
        with opener.open('https://pypi.org/pypi/'+package+'/'+version+'/json',timeout=45) as response:metadata=json.load(response)
        candidates=[]
        for item in metadata['urls']:
            if not item['filename'].endswith('.whl'):continue
            ranks=[order[tag] for tag in parse_wheel_filename(item['filename'])[3] if tag in order]
            if ranks:candidates.append((min(ranks),item))
        item=min(candidates,key=lambda pair:pair[0])[1]
        records.append({'url':item['url'],'sha256':item['digests']['sha256'],'filename':item['filename']})
    def digest(path):
        with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    def download(record):
        target=folder/record['filename'];partial=target.with_suffix(target.suffix+'.part')
        if target.exists() and digest(target)==record['sha256']:return target
        for attempt in range(16):
            offset=partial.stat().st_size if partial.exists() else 0
            try:
                request=urllib.request.Request(record['url'],headers={'Range':f'bytes={offset}-'} if offset else {})
                with opener.open(request,timeout=90) as response:
                    if offset and response.status==206:
                        match=re.match(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
                        if not match or int(match[1])!=offset:raise RuntimeError('Unexpected official wheel byte range')
                    elif response.status==200:offset=0
                    else:raise RuntimeError('Unexpected official wheel response')
                    checkpoint=offset//(256<<20)
                    with partial.open('ab' if offset else 'wb') as stream:
                        while block:=response.read(256<<10):
                            stream.write(block);offset+=len(block)
                            if offset//(256<<20)>checkpoint:
                                checkpoint=offset//(256<<20);print(record['filename'],round(offset/2**20),'MiB',flush=True)
                if digest(partial)!=record['sha256']:raise RuntimeError('Incomplete wheel or publisher SHA256 mismatch')
                partial.replace(target);print('VERIFIED',target.name,flush=True);return target
            except (OSError,RuntimeError) as error:
                if attempt==15:raise
                print('Retry',target.name,type(error).__name__,flush=True);time.sleep(min(20,2**attempt))
    with ThreadPoolExecutor(max_workers=2) as pool:paths=list(pool.map(download,records))
    target=ROOT/'.tooling/tensorrt-sdk';env['PIP_CONFIG_FILE']=os.devnull
    temporary=ROOT/'.runtime/tensorrt-pip-temp';temporary.mkdir(exist_ok=True)
    env['TEMP']=str(temporary);env['TMP']=str(temporary)
    installer=Path(sys.base_prefix)/('python.exe' if os.name=='nt' else 'bin/python3')
    result=subprocess.run([str(installer),'-m','pip','install','--disable-pip-version-check','--no-index','--no-deps','--upgrade','--target',str(target),*[str(path) for path in paths]],env=env)
    if result.returncode:return result.returncode
    (ROOT/'.runtime/tensorrt-native-source.json').write_text(json.dumps({'wheels':records,'install_target':str(target),'scope':'isolated SDK only; application packages unchanged'},indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
