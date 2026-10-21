from pathlib import Path
import argparse,hashlib,json,random,sys,os
import yaml
root=Path(__file__).resolve().parents[1];platform_root=Path(os.getenv('GUANYE_PLATFORM_ROOT',str(root))).resolve();out=root/'datasets';out.mkdir(exist_ok=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['snapshot','audit','split']);a=p.parse_args();params=yaml.safe_load((root/'params.yaml').read_text())
    if a.mode=='snapshot':
        import psycopg
        from psycopg.rows import dict_row
        sys.path.insert(0,str(platform_root/'integrations'));from platform_bridge import read_config
        import uuid;project=uuid.UUID(str(params.get('project_id')));cfg=read_config(platform_root)
        with psycopg.connect(host='127.0.0.1',port=15432,dbname='guanye',user='guanye',password=cfg['POSTGRES_PASSWORD'],row_factory=dict_row) as db:
            rows=db.execute("SELECT id::text,sha256,label,review_status,object_key FROM platform_samples WHERE project_id=%s ORDER BY id",(project,)).fetchall()
        if not rows:raise ValueError('Project has no samples. Do not create an empty evaluation dataset.')
        (out/'manifest.jsonl').write_text('\n'.join(json.dumps(dict(r),ensure_ascii=False,sort_keys=True) for r in rows)+'\n',encoding='utf-8');return
    rows=[json.loads(line) for line in (out/'manifest.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if a.mode=='audit':
        from collections import Counter
        counts=Counter(r['sha256'] for r in rows);result={'samples':len(rows),'duplicate_images':sum(v-1 for v in counts.values()),'confirmed':sum(r['review_status']=='confirmed' for r in rows),'manifest_sha256':hashlib.sha256((out/'manifest.jsonl').read_bytes()).hexdigest()};(out/'audit.json').write_text(json.dumps(result,indent=2));return
    groups={}
    for row in rows:groups.setdefault(row['sha256'],[]).append(row['id'])
    keys=sorted(groups);random.Random(params['seed']).shuffle(keys);cut=max(1,int(len(keys)*.8));train=keys[:cut];test=keys[cut:]
    if not test:raise ValueError('At least two independent images are required for a split')
    result={'seed':params['seed'],'train':[key for sha in train for key in groups[sha]],'test':[key for sha in test for key in groups[sha]],'grouped_by':'exact image sha256; no duplicate can cross splits'}
    (out/'splits.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
