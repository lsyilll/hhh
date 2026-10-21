"""Isolated runtime for newer research models; never mutates AgriChat's environment."""
import gc,hashlib,hmac,io,json,os,threading,uuid
from pathlib import Path
import numpy as np
from fastapi import FastAPI,HTTPException,Request
from pydantic import BaseModel,Field
from PIL import Image,ImageOps
from worker.common import s3,bucket,connection
from model_service.providers import MODEL_REPOS,encoder_name,weights_ready
from model_service.runtime import ResearchRuntime
root=Path(os.getenv('MODEL_ROOT','/models'));device=os.getenv('MODEL_DEVICE','cpu');lock=threading.Lock();active={}
runtime=ResearchRuntime(root,device);active=runtime.active
app=FastAPI(docs_url=None,redoc_url=None,openapi_url=None)
if os.getenv('OTEL_ENABLED')=='1':
    from telemetry import configure
    configure(app,'guanye-research-models')
paths=MODEL_REPOS
class Job(BaseModel):
    task_id:uuid.UUID
    owner_id:str=Field(min_length=1,max_length=64)
    kind:str
    args:dict={}
    samples:list[dict]=Field(min_length=1,max_length=100)
def installed(name):return weights_ready(root,name)
def require_auth(request):
    expected='Bearer '+os.environ['MODEL_SERVICE_SECRET']
    if not hmac.compare_digest(request.headers.get('authorization',''),expected):raise HTTPException(401,'Service authentication required')
@app.get('/health')
def health():return {'ready':True,'device':device,'active_model':active.get('name'),'weights':{k:installed(k) for k in paths},'defaults':{'encoder':'dinov2b','segmenter':'sam2'},'segmentation_prompts':{'sam2':['box'],'sam3':['text','box']}}
def load(name):
    return runtime.load(name)
def photo(row):
    key=row['object_key']
    if not key.startswith('users/'+row['owner_id']+'/'):raise ValueError('Object ownership mismatch')
    data=s3().get_object(Bucket=bucket(),Key=key)['Body'].read(20*1024*1024+1)
    if len(data)>20*1024*1024:raise ValueError('Image exceeds 20MB')
    with Image.open(io.BytesIO(data)) as src:
        if src.width*src.height>40_000_000:raise ValueError('Image exceeds 40MP')
        image=ImageOps.exif_transpose(src).convert('RGB');image.thumbnail((1280,1280));return image.copy()
def store(job,name,data,mime):
    key=f'users/{job.owner_id}/tasks/{job.task_id}/{name}';s3().put_object(Bucket=bucket(),Key=key,Body=data,ContentType=mime);return key
def embedding(images,name):
    return runtime.embedding(images,name)
def visual_features(job):
    import faiss
    name=encoder_name(job.args if job.kind!='dinov3' else {'model':job.args.get('model','dinov3b')})
    if job.kind=='dinov3' and name not in ('dinov3b','dinov3l'):raise ValueError('The legacy dinov3 task requires an actual DINOv3 encoder; use features for DINOv2')
    features=embedding([photo(row) for row in job.samples],name);index=faiss.IndexFlatIP(features.shape[1]);index.add(features)
    buffer=io.BytesIO();np.save(buffer,features,allow_pickle=False)
    features_key=store(job,'features.npy',buffer.getvalue(),'application/octet-stream')
    index_key=store(job,'index.faiss',faiss.serialize_index(index).tobytes(),'application/octet-stream')
    ids=[row['id'] for row in job.samples];ids_key=store(job,'index-ids.json',json.dumps(ids).encode(),'application/json')
    scores,indices=index.search(features,min(len(features),4))
    return {'encoder':name,'count':len(features),'dimension':features.shape[1],'features':features_key,'faiss':index_key,'ids':ids_key,
            'neighbors':[{'sample_id':ids[i],'matches':[{'sample_id':ids[int(j)],'cosine_similarity':round(float(s),4)} for s,j in zip(scores[i],indices[i]) if j>=0 and j!=i]} for i in range(len(ids))],
            'note':'Visual cosine similarity is not a disease probability.'}
def segment(job):
    name=job.args.get('segmenter','sam2');results=[];prompt=str(job.args.get('prompt','leaf'))[:100];box=job.args.get('box')
    for row in job.samples[:10]:
        image=photo(row);masks,scores=runtime.segmentation(image,name,box,prompt);entries=[]
        for i,mask in enumerate(masks[:20]):
            png=io.BytesIO();Image.fromarray((mask*255).astype(np.uint8)).save(png,format='PNG');key=store(job,f'{row["id"]}-mask-{i}.png',png.getvalue(),'image/png')
            entries.append({'mask':key,'segmentation_score':float(scores[i])})
        results.append({'sample_id':row['id'],'candidates':entries,'human_review_required':True})
    return {'model':name,'repository':MODEL_REPOS.get(name),'prompt_type':'box' if name=='sam2' else 'text_and_optional_box',
            'prompt':prompt if name=='sam3' else None,'box':box,'samples':results,'note':'Segmentation scores are mask quality estimates, not disease probabilities.'}
def report(job):
    import torch
    limit=job.args.get('max_new_tokens',256)
    if isinstance(limit,bool) or not isinstance(limit,int) or not 32<=limit<=512:raise ValueError('max_new_tokens must be an integer from 32 to 512')
    model,processor=load('qwen');rows=job.samples[:2];evidence=[{'id':'E'+str(i+1),'sample_id':r['id'],'model_answer':r['model_answer'],'human_label':r['label'],'human_status':r['review_status'],'notes':r['notes'],'quality':r['quality']} for i,r in enumerate(rows)]
    text='请用中文整理植物观察报告。只依据附图及下面记录，分别写可见事实、已有模型意见、人工意见、相互冲突和待核实信息。每条结论引用 E 编号，不得把模型意见写成已证实事实，不编造来源或检测结果。证据：'+json.dumps(evidence,ensure_ascii=False)
    images=[photo(r) for r in rows]
    for image in images:image.thumbnail((512,512))
    contents=[{'type':'image','image':image} for image in images]+[{'type':'text','text':text}]
    inputs=processor.apply_chat_template([{'role':'user','content':contents}],tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',enable_thinking=False).to(device)
    with torch.inference_mode():tokens=model.generate(**inputs,max_new_tokens=limit,do_sample=False)
    answer=processor.decode(tokens[0][inputs['input_ids'].shape[-1]:],skip_special_tokens=True)
    key=store(job,'evidence-report.md',answer.encode('utf-8'),'text/markdown; charset=utf-8')
    return {'model':'Qwen3.5-4B','report':key,'text':answer,'evidence':evidence,'human_review_required':True,'max_new_tokens':limit,'output_tokens':int(tokens.shape[-1]-inputs['input_ids'].shape[-1]),'generation_limit_reached':bool(tokens.shape[-1]-inputs['input_ids'].shape[-1]>=limit)}
def experiment(job):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
    from collections import Counter
    import mlflow
    rows=[r for r in job.samples if r['review_status']=='confirmed' and r['label']]
    counts=Counter(r['label'] for r in rows)
    if len(counts)<2 or min(counts.values())<5:raise ValueError('At least 2 human-confirmed classes, each with 5 independent samples, are required')
    # Exact duplicate source objects cannot cross training and test partitions.
    hashes=[hashlib.sha256(s3().get_object(Bucket=bucket(),Key=r['object_key'])['Body'].read()).hexdigest() for r in rows]
    if len(set(hashes))!=len(hashes):raise ValueError('Remove duplicate images before evaluating to prevent train/test leakage')
    name=encoder_name(job.args);x=embedding([photo(r) for r in rows],name);y=np.array([r['label'] for r in rows]);seed=int(job.args.get('seed',42))
    train,test=train_test_split(np.arange(len(rows)),test_size=max(len(counts),int(len(rows)*.25)),stratify=y,random_state=seed)
    classifier=LogisticRegression(max_iter=1000,random_state=seed).fit(x[train],y[train]);predicted=classifier.predict(x[test]);labels=sorted(counts)
    version=hashlib.sha256(json.dumps(sorted(zip(hashes,y.tolist()))).encode()).hexdigest()
    metrics={'accuracy':float(accuracy_score(y[test],predicted)),'macro_f1':float(f1_score(y[test],predicted,average='macro')),'test_samples':len(test)}
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI']);mlflow.set_experiment('guanye-'+str(job.samples[0]['project_id']))
    with mlflow.start_run() as run:
        mlflow.log_params({'backbone':name,'dataset_sha256':version,'seed':seed,'train':len(train),'test':len(test)});mlflow.log_metrics(metrics)
        mlflow.log_dict({'labels':labels,'matrix':confusion_matrix(y[test],predicted,labels=labels).tolist()},'confusion.json')
        from model_service.validate_onnx import export_classifier,export_numeric_head
        onnx_blob,consistency=export_classifier(classifier,x[test]);metrics.update(consistency)
        key=store(job,'classifier.onnx',onnx_blob,'application/onnx');mlflow.log_dict(metrics,'metrics.json')
        numeric=store(job,'numeric-head.onnx',export_numeric_head(classifier,x[test]),'application/onnx');sample_buffer=io.BytesIO();np.save(sample_buffer,x[test],allow_pickle=False);test_key=store(job,'test-features.npy',sample_buffer.getvalue(),'application/octet-stream')
        with connection() as db:db.execute('INSERT INTO model_experiments VALUES(%s,%s,%s,%s,%s,%s::jsonb,%s,%s,now())',(uuid.uuid4(),uuid.UUID(job.samples[0]['project_id']),job.owner_id,name,version,json.dumps(metrics),key,run.info.run_id))
        return {'model':name,'dataset_version':version,'metrics':metrics,'classifier_onnx':key,'numeric_head_onnx':numeric,'test_features':test_key,'mlflow_run':run.info.run_id,'labels':labels}
@app.post('/run')
def run(job:Job,request:Request):
    require_auth(request)
    functions={'features':visual_features,'dinov3':visual_features,'segment':segment,'evidence_report':report,'experiment':experiment}
    if job.kind not in functions:raise HTTPException(400,'Unknown model task')
    try:
        with lock:return functions[job.kind](job)
    except ValueError as error:raise HTTPException(409,str(error)) from None
