from pathlib import Path
import urllib.request,hashlib,json,zipfile,concurrent.futures,xml.etree.ElementTree as ET
root=Path(__file__).resolve().parents[1];dest=root/'.tooling';dest.mkdir(parents=True,exist_ok=True)
def fetch(url):
    return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Guanye-local-setup/1.0'}),timeout=120).read()
def archive(name,url,checksum_url,algorithm):
    try:
        path=dest/name
        check=fetch(checksum_url).decode().strip().split()[0].lower()
        if not path.exists():path.write_bytes(fetch(url))
        digest=hashlib.new(algorithm,path.read_bytes()).hexdigest();assert digest==check,(name,'checksum mismatch')
        with zipfile.ZipFile(path) as package:package.extractall(dest)
        print(name,'verified and extracted',flush=True)
        return {'name':name,'sha':digest,'source':url,'status':'ready'}
    except Exception as error:
        print(name,type(error).__name__,str(error),flush=True);return {'name':name,'status':'failed','error':str(error)}
jobs=[('jdk.zip','https://aka.ms/download-jdk/microsoft-jdk-21.0.12.1-windows-x64.zip','https://aka.ms/download-jdk/microsoft-jdk-21.0.12.1-windows-x64.zip.sha256sum.txt','sha256'),
      ('maven.zip','https://repo.maven.apache.org/maven2/org/apache/maven/apache-maven/3.9.11/apache-maven-3.9.11-bin.zip','https://repo.maven.apache.org/maven2/org/apache/maven/apache-maven/3.9.11/apache-maven-3.9.11-bin.zip.sha512','sha512')]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:report=list(pool.map(lambda args:archive(*args),jobs))
(dest/'download-check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
