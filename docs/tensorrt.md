


inference

```python
def do_inference_v2_time(self, context, bindings, inputs, outputs, stream):
        satrt_time = time.time()
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        stream.synchronize()
        copy_time = time.time()

        # Run inference.
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        stream.synchronize()
        execute_time = time.time()

        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        infer_time = time.time()

        cp_time = (copy_time - satrt_time) * 1000 + (infer_time - execute_time) * 1000
        enqune_time = (execute_time - copy_time) * 1000
        all_time = (infer_time - satrt_time) * 1000
        return  cp_time, enqune_time, all_time
```