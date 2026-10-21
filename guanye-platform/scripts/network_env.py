"""Reuse an already enabled local Windows proxy for a child command only."""
import os
import socket
from urllib.parse import urlsplit

def windows_download_env(source=None):
    env=dict(os.environ if source is None else source)
    if os.name!='nt' or env.get('HTTPS_PROXY') or env.get('https_proxy'):
        return env,None
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,r'Software\Microsoft\Windows\CurrentVersion\Internet Settings') as key:
            if not winreg.QueryValueEx(key,'ProxyEnable')[0]:return env,None
            server=winreg.QueryValueEx(key,'ProxyServer')[0]
        if '=' in server:
            mapping=dict(part.split('=',1) for part in server.split(';') if '=' in part)
            server=mapping.get('https') or mapping.get('http')
        if not server:return env,None
        url=server if '://' in server else 'http://'+server
        parsed=urlsplit(url)
        if parsed.scheme not in ('http','https') or parsed.hostname not in ('127.0.0.1','localhost','::1'):
            return env,None
        with socket.create_connection((parsed.hostname,parsed.port or 80),timeout=1):pass
        env.setdefault('HTTP_PROXY',url);env.setdefault('HTTPS_PROXY',url)
        bypass=env.get('NO_PROXY',env.get('no_proxy',''))
        env['NO_PROXY']=','.join(filter(None,[bypass,'localhost','127.0.0.1','::1']))
        return env,f'{parsed.hostname}:{parsed.port or 80}'
    except (OSError,ValueError,TypeError):
        return env,None
