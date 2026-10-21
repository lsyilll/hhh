import json
import math
import pytest
from model_service.providers import box_pixels, encoder_name, normalized_box, weights_ready

def test_weights_do_not_become_ready_from_config_alone(tmp_path):
    folder=tmp_path/'dinov2b';folder.mkdir()
    (folder/'config.json').write_text('{}');(folder/'preprocessor_config.json').write_text('{}')
    (folder/'model.safetensors.part').write_bytes(b'incomplete')
    assert not weights_ready(tmp_path,'dinov2b')
    (folder/'model.safetensors').write_bytes(b'checkpoint')
    assert weights_ready(tmp_path,'dinov2b')

def test_all_checkpoint_shards_must_be_present_and_local(tmp_path):
    folder=tmp_path/'qwen';folder.mkdir()
    for name in ['config.json','preprocessor_config.json']:(folder/name).write_text('{}')
    index=folder/'model.safetensors.index.json'
    index.write_text(json.dumps({'weight_map':{'a':'part-1.safetensors','b':'part-2.safetensors'}}))
    (folder/'part-1.safetensors').write_bytes(b'one')
    assert not weights_ready(tmp_path,'qwen')
    (folder/'part-2.safetensors').write_bytes(b'two')
    assert weights_ready(tmp_path,'qwen')
    index.write_text(json.dumps({'weight_map':{'a':'../outside.safetensors'}}))
    assert not weights_ready(tmp_path,'qwen')

@pytest.mark.parametrize('box',[None,[.5,.5,0,.2],[.1,.5,.9,.2],[.5,.5,math.nan,.2],[True,.5,.2,.2],[.5,.5,1.1,.2]])
def test_invalid_region_cannot_reach_segmentation(box):
    with pytest.raises(ValueError):normalized_box(box)

def test_normalized_region_tracks_actual_image_dimensions():
    assert box_pixels([.5,.4,.4,.6],200,100)==pytest.approx([60,10,140,70])

def test_requested_model_identity_is_preserved():
    assert encoder_name({})=='dinov2b'
    assert encoder_name({'model':'dinov3l'})=='dinov3l'
    with pytest.raises(ValueError):encoder_name({'model':'sam2'})
