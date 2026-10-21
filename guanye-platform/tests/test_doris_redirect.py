import pytest
from worker.analytics_sink import stream_load_target

ORIGINAL='http://doris:8030/api/guanye/ods_events/_stream_load'

@pytest.fixture(autouse=True)
def dns(monkeypatch):
    monkeypatch.setattr('worker.analytics_sink.socket.getaddrinfo',lambda *args:[(None,None,None,None,('172.18.0.9',0))])

def test_doris_real_redirect_discards_url_credentials():
    target=stream_load_target('http://root:discarded@127.0.0.1:8040/api/guanye/ods_events/_stream_load',ORIGINAL,'doris')
    assert target=='http://doris:8040/api/guanye/ods_events/_stream_load'
    assert 'discarded' not in target and '@' not in target

@pytest.mark.parametrize('location',[
    'http://attacker.invalid:8040/api/guanye/ods_events/_stream_load',
    'http://172.18.0.88:8040/api/guanye/ods_events/_stream_load',
    'http://doris:8080/api/guanye/ods_events/_stream_load',
    'http://doris:8040/api/other/secret/_stream_load',
    'https://doris:8040/api/guanye/ods_events/_stream_load',
])
def test_unexpected_service_redirect_rejected(location):
    with pytest.raises(ValueError):stream_load_target(location,ORIGINAL,'doris')


def test_stream_load_label_accepts_dotted_kafka_topics_without_collisions():
    import re
    from worker.analytics_sink import stream_load_label
    samples=[stream_load_label(topic,0,9) for topic in ['guanye.events','guanye_events','guanye.metrics.minute']]
    assert len(set(samples))==3
    assert all(re.fullmatch(r'[\-_A-Za-z0-9:]{1,128}',label) for label in samples)
    assert stream_load_label('guanye.events',0,9)==samples[0]
