"""Actual queued, image-grounded Qwen evidence report with scoped S3 readback."""
import time
import httpx

def check(**context):
    response=httpx.get('http://127.0.0.1:18082/health',trust_env=False,timeout=10)
    response.raise_for_status();assert response.json()['weights']['qwen']
    start=time.monotonic()
    row=context['task']('evidence_report',context['sample'],{'max_new_tokens':96})
    assert row['model']=='Qwen3.5-4B' and row['human_review_required']
    assert row['report'].startswith(context['prefix']) and len(row['text'].strip())>10
    assert row['evidence'][0]['id']=='E1' and row['evidence'][0]['sample_id']==context['sample']
    assert row['max_new_tokens']==96 and 0<row['output_tokens']<=96
    assert context['store'].get_object(Bucket=context['bucket'],Key=row['report'])['Body'].read().decode('utf-8')==row['text']
    return {'status':'pass','model':row['model'],'queued_evidence_report':'pass','s3_utf8_report_readback':'pass','seconds':round(time.monotonic()-start,2),'output_tokens':row['output_tokens'],'generation_limit_reached':row['generation_limit_reached'],'citation_E1_present':'E1' in row['text'],'preview':row['text'][:400],'scope':'actual public plant fixture; short report-generation functionality check; content requires human review; no diagnostic accuracy claim'}
