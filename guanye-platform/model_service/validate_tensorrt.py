"""Run a numeric ONNX head with TensorRT and compare it with ONNX Runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sysconfig
import time
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    for name in ('onnx','features','engine','report'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    handles=[]
    if os.name=='nt':
        site=Path(os.getenv('GUANYE_TENSORRT_SDK',sysconfig.get_paths()['purelib']))
        for folder in [site/'nvidia/cuda_runtime/bin',site/'nvidia/cuda_runtime/lib',site/'tensorrt_libs']:
            if folder.is_dir():handles.append(os.add_dll_directory(str(folder)))
    try:import tensorrt as trt
    except ModuleNotFoundError as error:
        if error.name!='tensorrt':raise
        import tensorrt_libs
        import tensorrt_bindings as trt
    import onnxruntime as ort
    from cuda.bindings import runtime as cuda

    def checked(result):
        error,*values=result
        if error!=cuda.cudaError_t.cudaSuccess:raise RuntimeError('CUDA runtime error '+str(error))
        return values[0] if len(values)==1 else values

    x=np.ascontiguousarray(np.load(args.features,allow_pickle=False),dtype=np.float32)
    if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('Expected a finite 2-D feature array')
    logger=trt.Logger(trt.Logger.WARNING);builder=trt.Builder(logger)
    network=builder.create_network(0);onnx_parser=trt.OnnxParser(network,logger)
    blob=Path(args.onnx).read_bytes()
    if not onnx_parser.parse(blob):raise RuntimeError('\n'.join(str(onnx_parser.get_error(i)) for i in range(onnx_parser.num_errors)))
    if network.num_inputs!=1 or network.num_outputs!=1:raise ValueError('Expected one numeric input and output; label/string graphs are not supported')
    name=network.get_input(0).name
    profile=builder.create_optimization_profile();profile.set_shape(name,x.shape,x.shape,x.shape)
    config=builder.create_builder_config();config.add_optimization_profile(profile)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,512<<20)
    if hasattr(trt.BuilderFlag,'TF32'):config.clear_flag(trt.BuilderFlag.TF32)
    serialized=builder.build_serialized_network(network,config)
    if serialized is None:raise RuntimeError('TensorRT engine build failed')
    Path(args.engine).write_bytes(bytes(serialized))
    runtime=trt.Runtime(logger);engine=runtime.deserialize_cuda_engine(serialized)
    context=engine.create_execution_context()
    if not context.set_input_shape(name,x.shape):raise RuntimeError('TensorRT rejected the feature shape')
    output_name=next(engine.get_tensor_name(i) for i in range(engine.num_io_tensors) if engine.get_tensor_mode(engine.get_tensor_name(i))==trt.TensorIOMode.OUTPUT)
    output=np.empty(tuple(context.get_tensor_shape(output_name)),dtype=trt.nptype(engine.get_tensor_dtype(output_name)))
    stream=checked(cuda.cudaStreamCreate());allocated=[]
    try:
        source=checked(cuda.cudaMalloc(x.nbytes));allocated.append(source)
        destination=checked(cuda.cudaMalloc(output.nbytes));allocated.append(destination)
        checked(cuda.cudaMemcpy(source,x.ctypes.data,x.nbytes,cuda.cudaMemcpyKind.cudaMemcpyHostToDevice))
        if not context.set_tensor_address(name,int(source)) or not context.set_tensor_address(output_name,int(destination)):
            raise RuntimeError('TensorRT tensor address registration failed')
        start=time.perf_counter()
        if not context.execute_async_v3(int(stream)):raise RuntimeError('TensorRT execution failed')
        checked(cuda.cudaStreamSynchronize(stream));elapsed=time.perf_counter()-start
        checked(cuda.cudaMemcpy(output.ctypes.data,destination,output.nbytes,cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost))
        reference=ort.InferenceSession(args.onnx,providers=['CPUExecutionProvider']).run(None,{name:x})[0]
        difference=float(np.max(np.abs(reference-output)))
        if not np.allclose(reference,output,atol=1e-4,rtol=1e-3):raise RuntimeError('TensorRT/ONNX Runtime output mismatch: '+str(difference))
        props=checked(cuda.cudaGetDeviceProperties(0))
        device_name=props.name.decode(errors='replace') if isinstance(props.name,bytes) else str(props.name)
        report={'status':'pass','passed':True,'tensorrt':trt.__version__,'onnxruntime':ort.__version__,'gpu':device_name,'cuda_runtime':checked(cuda.cudaRuntimeGetVersion()),'max_abs_error':difference,'samples':len(x),'features':x.shape[1],'onnx_sha256':hashlib.sha256(blob).hexdigest(),'one_execution_seconds':elapsed,'scope':'numeric-head backend consistency only; fixture input is synthetic; no disease accuracy or full AgriChat acceleration claim'}
        Path(args.report).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False,indent=2))
    finally:
        for pointer in allocated:checked(cuda.cudaFree(pointer))
        checked(cuda.cudaStreamDestroy(stream))


if __name__=='__main__':main()
