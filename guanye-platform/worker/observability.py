import functools,os,time
enabled=os.getenv('OTEL_ENABLED','0')=='1'
if enabled:
    from telemetry import configure
    configure(None,'guanye-worker')
    from opentelemetry import metrics,trace
    meter=metrics.get_meter('guanye-worker');duration=meter.create_histogram('task',unit='s',description='Actual worker attempt duration');count=meter.create_counter('tasks',description='Actual task execution attempts');tracer=trace.get_tracer('guanye-worker')
def observed(function):
    @functools.wraps(function)
    def wrapper(*args,**kwargs):
        if not enabled:return function(*args,**kwargs)
        started=time.monotonic();status='failed';duplicate=False
        try:
            with tracer.start_as_current_span('guanye.task'):
                result=function(*args,**kwargs);status='complete';duplicate=isinstance(result,dict) and bool(result.get('idempotent'));return result
        finally:
            if not duplicate:duration.record(time.monotonic()-started,{'status':status});count.add(1,{'status':status})
    return wrapper
