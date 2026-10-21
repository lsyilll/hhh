"""Transport tests with controlled byte fixtures, never real model inference."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from urllib.parse import unquote,urlsplit

import pytest


def setup_transport(tmp_path,monkeypatch,mode):
    script=Path(__file__).resolve().parents[1]/'scripts/download_models.py'
    spec=importlib.util.spec_from_file_location('guanye_transport_test',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(module.time,'sleep',lambda seconds:None)
    payload=(bytes(range(256))*4096)
    files={'config.json':b'{"model_type":"dinov2"}','preprocessor_config.json':b'{}','model.safetensors':payload}
    metadata={'sha':'fixed-commit','gated':False,'siblings':[{'rfilename':name,'size':len(data),'lfs':{'sha256':hashlib.sha256(data).hexdigest()}} for name,data in files.items()]}
    requests=[];interrupted=[False]
    class Response(io.BytesIO):
        def __init__(self,data,status=200,headers=None,interrupt=False):
            super().__init__(data);self.status=status;self.headers=headers or {};self.interrupt=interrupt;self.reads=0
        def read(self,n=-1):
            self.reads+=1
            if self.interrupt and self.reads==2:raise OSError('controlled transport interruption')
            return super().read(n)
    class Opener:
        def open(self,request,timeout=None):
            parsed=urlsplit(request.full_url)
            if '/api/models/' in parsed.path:return Response(json.dumps(metadata).encode())
            name=unquote(parsed.path.rsplit('/',1)[-1]);data=files[name];offset=0
            header=request.get_header('Range')
            if name=='model.safetensors':requests.append(header)
            if header and mode!='ignore':
                offset=int(header.split('=')[1].split('-')[0])
                start=0 if mode=='wrong_range' else offset
                headers={'Content-Range':f'bytes {start}-{len(data)-1}/{len(data)}'}
                return Response(data[start:],206,headers)
            interrupt=mode=='interrupt' and name=='model.safetensors' and not interrupted[0]
            if interrupt:interrupted[0]=True
            return Response(data,interrupt=interrupt)
    monkeypatch.setattr(module,'OPENER',Opener())
    folder=tmp_path/'.runtime/models/dinov2b';folder.mkdir(parents=True)
    return module,payload,folder,requests


def test_resumes_verified_partial_checkpoint(tmp_path,monkeypatch):
    module,data,folder,requests=setup_transport(tmp_path,monkeypatch,'resume')
    (folder/'model.safetensors.part').write_bytes(data[:1000])
    record=module.download_model('dinov2b','fixed-commit')
    assert (folder/'model.safetensors').read_bytes()==data
    assert 'bytes=1000-' in requests and not (folder/'model.safetensors.part').exists()
    assert next(r for r in record['files'] if r['path']=='model.safetensors')['publisher_sha256']==hashlib.sha256(data).hexdigest()


def test_ignored_range_replaces_partial_instead_of_appending(tmp_path,monkeypatch):
    module,data,folder,requests=setup_transport(tmp_path,monkeypatch,'ignore')
    (folder/'model.safetensors.part').write_bytes(data[:1000])
    module.download_model('dinov2b','fixed-commit')
    assert (folder/'model.safetensors').read_bytes()==data


def test_mismatched_range_never_appends_or_publishes(tmp_path,monkeypatch):
    module,data,folder,requests=setup_transport(tmp_path,monkeypatch,'wrong_range')
    partial=folder/'model.safetensors.part';partial.write_bytes(data[:1000])
    with pytest.raises(RuntimeError,match='Unexpected download range'):
        module.download_model('dinov2b','fixed-commit')
    assert partial.read_bytes()==data[:1000] and not (folder/'model.safetensors').exists()


def test_interrupted_stream_retries_from_saved_bytes(tmp_path,monkeypatch):
    module,data,folder,requests=setup_transport(tmp_path,monkeypatch,'interrupt')
    module.download_model('dinov2b','fixed-commit')
    assert (folder/'model.safetensors').read_bytes()==data
    assert 'bytes=262144-' in requests
