import numpy as np
from sklearn.linear_model import LogisticRegression
from model_service.validate_onnx import export_classifier
def test_exported_classifier_preserves_probabilities():
    # Verifies the export contract on controlled numerical inputs, not plant recognition accuracy.
    rng=np.random.default_rng(19);features=rng.normal(size=(40,8)).astype(np.float32);labels=(features[:,0]+features[:,1]>.1).astype(int)
    classifier=LogisticRegression().fit(features[:30],labels[:30]);blob,result=export_classifier(classifier,features[30:])
    assert len(blob)>100 and result['onnx_max_abs_error']<1e-5 and result['onnx_checked_samples']==10
