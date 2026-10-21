"""Temporary HTTP adapter using the production platform/device route module."""
from pathlib import Path
import types
from fastapi import FastAPI,HTTPException,Request
from fastapi.responses import PlainTextResponse
import uvicorn
from integrations.platform_bridge import install_platform_routes

ROOT=Path(__file__).resolve().parents[1]
app=FastAPI(docs_url=None,redoc_url=None,openapi_url=None)
def require(request):raise HTTPException(401,'Temporary TLS verification adapter has no user login')
async def body(request):return await request.json()
install_platform_routes(app,types.SimpleNamespace(ROOT=ROOT.parent),require,body)
@app.get('/')
def home():return PlainTextResponse('Temporary Guanye TLS integration adapter')
if __name__=='__main__':uvicorn.run(app,host='127.0.0.1',port=7860,log_level='warning')
