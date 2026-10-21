"""Opt-in operational tracing: route/timing only; no request bodies, faces or audio."""
import os
def configure(app,name):
    if os.getenv('OTEL_ENABLED','0')!='1':return False
    from opentelemetry import trace,metrics
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    endpoint=os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT','http://127.0.0.1:14318').rstrip('/')
    resource=Resource.create({'service.name':name});provider=TracerProvider(resource=resource);provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint+'/v1/traces')));trace.set_tracer_provider(provider)
    metrics.set_meter_provider(MeterProvider(resource=resource,metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=endpoint+'/v1/metrics'))]))
    if app is not None:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app,excluded_urls='.*auth.*,.*/voice,.*device-certificate.*')
    return True
