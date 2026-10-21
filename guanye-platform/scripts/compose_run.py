"""Run Compose normally and retain its actual output for local troubleshooting."""
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from network_env import windows_download_env

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--docker',required=True)
    parser.add_argument('docker_args',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    if not args.docker_args:
        parser.error('Docker arguments are required')
    folder=ROOT/'.runtime';folder.mkdir(exist_ok=True)
    archive=folder/('compose-start-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'.log')
    latest=folder/'compose-last-start.log'
    print('Docker output log: '+str(latest),flush=True)
    env,proxy=windows_download_env()
    note=('Using the existing Windows proxy for this command only: '+proxy+'\n') if proxy else ''
    if note:print(note,end='',flush=True)
    with archive.open('wb') as history,latest.open('wb') as live:
        history.write(note.encode('utf-8'));live.write(note.encode('utf-8'));live.flush()
        process=subprocess.Popen([args.docker,*args.docker_args],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        for block in iter(lambda:process.stdout.read1(8192),b''):
            history.write(block);history.flush();live.write(block);live.flush()
            sys.stdout.buffer.write(block);sys.stdout.buffer.flush()
        code=process.wait()
        note=f'\nDocker Compose exit code: {code}\n'.encode('utf-8')
        history.write(note);live.write(note)
    print('Docker Compose exit code:',code,flush=True)
    return code

if __name__=='__main__':sys.exit(main())
