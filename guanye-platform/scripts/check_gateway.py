"""Own and close every temporary process used for gateway validation."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import httpx

ROOT=Path(__file__).resolve().parents[1]

def main():
    runtime=ROOT/'.runtime/extended';runtime.mkdir(exist_ok=True)
    for port in (7860,8443,8444):
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1',port))==0:raise RuntimeError(f'Port {port} is in use; existing services will not be replaced')
    nginx=next((ROOT/'.tooling/nginx').glob('*/nginx.exe'))
    env=os.environ.copy();env.update(PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8')
    subprocess.run([sys.executable,str(ROOT/'scripts/make_gateway.py')],cwd=ROOT,env=env,check=True)
    arguments=[str(nginx),'-p',nginx.parent.as_posix()+'/', '-c',(ROOT/'.runtime/nginx.conf').as_posix()]
    subprocess.run(arguments+['-t'],env=env,check=True)
    adapter=None;proxy=None
    with (runtime/'gateway.log').open('wb') as log:
        try:
            adapter=subprocess.Popen([sys.executable,str(ROOT/'scripts/gateway_qa_adapter.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            deadline=time.monotonic()+30
            while True:
                try:
                    if httpx.get('http://127.0.0.1:7860/',trust_env=False,timeout=2).status_code==200:break
                except httpx.HTTPError:pass
                if time.monotonic()>deadline or adapter.poll() is not None:raise RuntimeError('Temporary gateway adapter failed to start')
                time.sleep(.25)
            proxy=subprocess.Popen(arguments+['-g','daemon off;'],cwd=nginx.parent,env=env,stdout=log,stderr=subprocess.STDOUT)
            time.sleep(1)
            if proxy.poll() is not None:raise RuntimeError('Nginx process failed to start; see gateway.log')
            result=subprocess.run([sys.executable,str(ROOT/'scripts/smoke_extended.py'),'gateway'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
            return result.returncode
        finally:
            if proxy is not None and proxy.poll() is None:
                subprocess.run(arguments+['-s','quit'],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=15)
                try:proxy.wait(timeout=15)
                except subprocess.TimeoutExpired:proxy.terminate();proxy.wait(timeout=5)
            if adapter is not None and adapter.poll() is None:adapter.terminate();adapter.wait(timeout=10)

if __name__=='__main__':raise SystemExit(main())
