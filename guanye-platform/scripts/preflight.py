from pathlib import Path
import json,os,platform,shutil,socket,subprocess,sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from model_service.providers import MODEL_REPOS,weights_ready
def run(args):
    try:
        p=subprocess.run(args,capture_output=True,timeout=15);raw=p.stdout+p.stderr
        text=raw.decode('utf-16-le',errors='replace') if raw.count(b'\0')>len(raw)//6 else raw.decode('utf-8',errors='replace')
        return {'exit_code':p.returncode,'output':text[:1000]}
    except (OSError,subprocess.TimeoutExpired) as e:return {'exit_code':None,'output':str(e)}
def main():
    docker=shutil.which('docker') or next((str(p) for p in [Path(os.environ.get('USERPROFILE','C:/Users/12203'))/'AppData/Local/Programs/DockerDesktop/resources/bin/docker.exe',Path('C:/Program Files/Docker/Docker/resources/bin/docker.exe')] if p.is_file()),str(root/'.tooling/docker/docker/docker.exe'));compose=root/'.tooling/docker-compose.exe'
    result={'python':sys.version.split()[0],'system':platform.platform(),'docker_engine':run([docker,'info','--format','{{.ServerVersion}}']),
            'compose':run([str(compose),'--project-directory',str(root),'--profile','*','config','--quiet']),
            'model_weights':{name:weights_ready(root/'.runtime/models',name) for name in MODEL_REPOS}}
    if os.name=='nt':
        result['wsl_version']=run([r'C:\Windows\System32\wsl.exe','--version'])
        result['wsl']=run([r'C:\Windows\System32\wsl.exe','--status'])
    report=root/'.runtime/preflight.json';report.parent.mkdir(exist_ok=True);report.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k not in ('docker_engine','wsl')},ensure_ascii=False,indent=2));print('Docker and WSL diagnostics saved locally. This check does not modify Windows.')
if __name__=='__main__':main()
