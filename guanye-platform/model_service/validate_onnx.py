import numpy as np
def export_classifier(classifier,test_features):
    import onnxruntime as ort
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    model=convert_sklearn(classifier,initial_types=[('features',FloatTensorType([None,test_features.shape[1]]))],options={id(classifier):{'zipmap':False}},target_opset=17)
    blob=model.SerializeToString();session=ort.InferenceSession(blob,providers=['CPUExecutionProvider']);values=session.run(None,{'features':test_features.astype(np.float32)})
    expected=classifier.predict_proba(test_features);error=float(np.max(np.abs(values[1]-expected)))
    if not np.allclose(values[1],expected,atol=1e-5,rtol=1e-4):raise ValueError('ONNX validation failed')
    return blob,{'onnx_max_abs_error':error,'onnx_checked_samples':len(test_features)}

def export_numeric_head(classifier,test_features):
    import io,torch
    head=torch.nn.Linear(test_features.shape[1],classifier.coef_.shape[0]).eval()
    with torch.no_grad():head.weight.copy_(torch.from_numpy(classifier.coef_.astype(np.float32)));head.bias.copy_(torch.from_numpy(classifier.intercept_.astype(np.float32)))
    output=io.BytesIO();torch.onnx.export(head,torch.from_numpy(test_features.astype(np.float32)),output,input_names=['features'],output_names=['logits'],opset_version=17,dynamo=False,dynamic_axes={'features':{0:'batch'},'logits':{0:'batch'}})
    return output.getvalue()
