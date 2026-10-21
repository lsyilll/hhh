"""Execute actual local model inference; no server, database, or cloud image upload."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image, ImageOps

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model_service.runtime import ResearchRuntime
from model_service.providers import weights_ready, MODEL_REPOS

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--image',type=Path,default=ROOT.parent/'test-images/tomato_healthy.jpg')
    parser.add_argument('--models',nargs='+',default=['dinov2b','dinov2l','sam2'])
    parser.add_argument('--output',type=Path,default=ROOT/'.runtime/public-model-check')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    import torch,faiss
    torch.set_num_threads(4)
    image=ImageOps.exif_transpose(Image.open(args.image)).convert('RGB')
    runtime=ResearchRuntime(ROOT/'.runtime/models','cpu')
    result={'device':'cpu','torch':torch.__version__,'image':str(args.image),'image_sha256':hashlib.sha256(args.image.read_bytes()).hexdigest(),'checks':{},
            'scope':'Real local inference and output contracts; not disease accuracy or end-to-end container validation.'}
    for name in args.models:
        if not weights_ready(runtime.root,name):
            result['checks'][name]={'status':'missing_weights'};continue
        started=time.monotonic();print('INFERENCE',name,flush=True)
        if name.startswith('dino'):
            vectors=runtime.embedding([image,image.copy()],name)
            assert vectors.shape[0]==2 and np.isfinite(vectors).all()
            assert np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5)
            cosine=float(vectors[0]@vectors[1]);assert cosine>.9999
            index=faiss.IndexFlatIP(vectors.shape[1]);index.add(vectors)
            scores,ids=index.search(vectors[:1],2)
            assert np.all(scores>.9999)
            result['checks'][name]={'status':'pass','repository':MODEL_REPOS[name],'dimension':vectors.shape[1],'same_image_cosine':cosine,'faiss_results':ids[0].tolist()}
        elif name=='sam2':
            box=[.5,.5,.9,.9];masks,scores=runtime.segmentation(image,'sam2',box=box)
            assert masks.shape[1:]==(image.height,image.width) and len(masks)==len(scores)>0
            assert np.isfinite(scores).all() and any(m.any() for m in masks)
            best=int(np.argmax(scores));mask=masks[best]
            Image.fromarray(mask.astype(np.uint8)*255).save(args.output/'sam2-mask.png')
            rgba=image.convert('RGBA');tint=Image.new('RGBA',image.size,(32,145,90,0));tint.putalpha(Image.fromarray(mask.astype(np.uint8)*100))
            Image.alpha_composite(rgba,tint).convert('RGB').save(args.output/'sam2-leaf-overlay.jpg',quality=93)
            result['checks'][name]={'status':'pass','repository':MODEL_REPOS[name],'candidates':len(masks),'mask_shape':list(masks.shape[1:]),'mask_quality_scores':[float(s) for s in scores],'box':box,'human_review_required':True}
        result['checks'][name]['seconds']=round(time.monotonic()-started,2)
        (args.output/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        runtime.clear();print('PASS',name,flush=True)
    (args.output/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True,indent=2),flush=True)

if __name__=='__main__':main()
