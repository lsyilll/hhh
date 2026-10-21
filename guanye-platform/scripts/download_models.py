"""Download official weights with pinned revisions and verified file digests."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request
import urllib.parse
import time
import re
import socket
import http.client
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model_service.providers import MODEL_REPOS, PUBLIC_MODELS, PUBLIC_REVISIONS, weights_ready

class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and urllib.parse.urlsplit(req.full_url).netloc != urllib.parse.urlsplit(newurl).netloc:
            redirected.remove_header('Authorization')
        return redirected

OPENER = urllib.request.build_opener(SafeRedirect())

def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def download_model(name, revision=None):
    repo = MODEL_REPOS[name]
    # Public choices never read an implicit login token. Gated models retain explicit authorization.
    token = False if name in PUBLIC_MODELS else os.getenv('HF_TOKEN')
    revision = revision or PUBLIC_REVISIONS.get(name, 'main')
    headers = {'User-Agent': 'Guanye-public-model-downloader/1.0'}
    if token:
        headers['Authorization'] = 'Bearer ' + token
    print('Resolving official source:', name, repo, flush=True)
    url = f'https://huggingface.co/api/models/{repo}/revision/{urllib.parse.quote(revision, safe="")}?blobs=true'
    with OPENER.open(urllib.request.Request(url, headers=headers), timeout=45) as response:
        info = json.load(response)
    folder = ROOT / '.runtime/models' / name
    folder.mkdir(parents=True, exist_ok=True)
    def wanted(filename):
        if name == 'sam3':
            return filename == 'sam3.pt' or filename.endswith(('.json', '.txt', '.md'))
        return filename.endswith(('.json', '.safetensors', '.txt', '.md', '.model', '.tiktoken', '.jinja', '.jinja'))
    files = [f for f in info['siblings'] if wanted(f['rfilename'])]
    def one(item):
        filename = item['rfilename']
        if filename.startswith('/') or '..' in Path(filename).parts or '\\' in filename:
            raise RuntimeError('Unexpected repository path')
        target = folder / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = item.get('lfs', {}).get('sha256')
        size = item.get('size')
        if target.is_file() and (size is None or target.stat().st_size == size):
            digest = sha256(target)
            if not expected or digest == expected:
                return {'path': filename, 'size': target.stat().st_size, 'sha256': digest, 'publisher_sha256': expected}
        url = f'https://huggingface.co/{repo}/resolve/{info["sha"]}/{filename}'
        partial = target.with_name(target.name + '.part')
        for attempt in range(8):
            offset=partial.stat().st_size if partial.exists() else 0
            if size is not None and offset>size:
                partial.unlink();offset=0
            try:
                if size is None or offset!=size:
                    request_headers=dict(headers)
                    if offset:request_headers['Range']='bytes='+str(offset)+'-'
                    request_url=url+'?download=true&guanye_retry='+str(attempt)
                    with OPENER.open(urllib.request.Request(request_url,headers=request_headers),timeout=60) as source:
                        if offset and source.status==206:
                            match=re.match(r'bytes (\d+)-(\d+)/(\d+)',source.headers.get('Content-Range',''))
                            if not match or int(match[1])!=offset:raise RuntimeError('Unexpected download range; refusing to append')
                        elif offset and source.status==200:
                            offset=0
                        elif offset:
                            raise RuntimeError('Server did not provide a valid resumed response')
                        print(name,filename,'resume MiB',round(offset/2**20,1),'attempt',attempt+1,flush=True)
                        with partial.open('ab' if offset else 'wb') as dest:
                            checkpoint=offset//(128*2**20)
                            while block:=source.read(256*1024):
                                dest.write(block);offset+=len(block)
                                if offset//(128*2**20)>checkpoint:
                                    checkpoint=offset//(128*2**20)
                                    print(name,filename,'MiB',round(offset/2**20,1),flush=True)
                if size is not None and partial.stat().st_size!=size:raise RuntimeError('Incomplete download: '+filename)
                digest=sha256(partial)
                if expected and digest!=expected:
                    partial.unlink();raise RuntimeError('Publisher SHA256 mismatch: '+filename)
                break
            except (OSError,RuntimeError,urllib.error.URLError,http.client.HTTPException) as error:
                if attempt==7:raise
                print(name,filename,'retry after',type(error).__name__,'retained bytes',partial.stat().st_size if partial.exists() else 0,flush=True)
                time.sleep(min(30,2**attempt))
        partial.replace(target)
        print(name, filename, target.stat().st_size, flush=True)
        return {'path': filename, 'size': target.stat().st_size, 'sha256': digest, 'publisher_sha256': expected}
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(one, files))
    record = {'repository': repo, 'revision': info['sha'], 'gated': info.get('gated'), 'license':info.get('cardData',{}).get('license'), 'files': records}
    (folder / 'guanye-source.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    if not weights_ready(ROOT / '.runtime/models', name):
        raise RuntimeError('Required model files are incomplete: ' + name)
    print('READY', name, repo, info['sha'], flush=True)
    return record

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=MODEL_REPOS)
    parser.add_argument('--revision')
    args = parser.parse_args()
    download_model(args.model, args.revision)

if __name__ == '__main__':
    main()
