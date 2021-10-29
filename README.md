# openvino_sample

## hello_query_device

```
> hello_query_device.exe

Available devices:
        Device: CPU
        Metrics:
                SUPPORTED_METRICS : [ AVAILABLE_DEVICES SUPPORTED_METRICS FULL_DEVICE_NAME OPTIMIZATION_CAPABILITIES SUPPORTED_CONFIG_KEYS RANGE_FOR_ASYNC_INFER_REQUESTS RANGE_FOR_STREAMS ]
                FULL_DEVICE_NAME : Intel(R) Core(TM) i7-8665U CPU @ 1.90GHz
                OPTIMIZATION_CAPABILITIES : [ FP32 FP16 INT8 BIN ]
                SUPPORTED_CONFIG_KEYS : [ CPU_BIND_THREAD CPU_THREADS_NUM CPU_THROUGHPUT_STREAMS DUMP_EXEC_GRAPH_AS_DOT DYN_BATCH_ENABLED DYN_BATCH_LIMIT ENFORCE_BF16 EXCLUSIVE_ASYNC_REQUESTS PERF_COUNT ]
                RANGE_FOR_ASYNC_INFER_REQUESTS : { 1, 1, 1 }
                RANGE_FOR_STREAMS : { 1, 8 }
        Default values for device configuration keys:
                CPU_BIND_THREAD : NUMA
                CPU_THREADS_NUM : 0
                CPU_THROUGHPUT_STREAMS : 1
                DUMP_EXEC_GRAPH_AS_DOT : ""
                DYN_BATCH_ENABLED : NO
                DYN_BATCH_LIMIT : 0
                ENFORCE_BF16 : NO
                EXCLUSIVE_ASYNC_REQUESTS : NO
                PERF_COUNT : NO

        Device: GPU
        Metrics:
                SUPPORTED_METRICS : [ AVAILABLE_DEVICES SUPPORTED_METRICS FULL_DEVICE_NAME OPTIMIZATION_CAPABILITIES SUPPORTED_CONFIG_KEYS RANGE_FOR_ASYNC_INFER_REQUESTS RANGE_FOR_STREAMS ]
                FULL_DEVICE_NAME : Intel(R) UHD Graphics 620 (iGPU)
                OPTIMIZATION_CAPABILITIES : [ FP32 BIN FP16 ]
                SUPPORTED_CONFIG_KEYS : [ CACHE_DIR CLDNN_ENABLE_FP16_FOR_QUANTIZED_MODELS CLDNN_GRAPH_DUMPS_DIR CLDNN_MEM_POOL CLDNN_NV12_TWO_INPUTS CLDNN_PLUGIN_PRIORITY CLDNN_PLUGIN_THROTTLE CLDNN_SOURCES_DUMPS_DIR CONFIG_FILE DEVICE_ID DUMP_KERNELS DYN_BATCH_ENABLED EXCLUSIVE_ASYNC_REQUESTS GPU_THROUGHPUT_STREAMS PERF_COUNT TUNING_FILE TUNING_MODE ]
                RANGE_FOR_ASYNC_INFER_REQUESTS : { 1, 2, 1 }
                RANGE_FOR_STREAMS : { 1, 2 }
        Default values for device configuration keys:
                CACHE_DIR : ""
                CLDNN_ENABLE_FP16_FOR_QUANTIZED_MODELS : YES
                CLDNN_GRAPH_DUMPS_DIR : ""
                CLDNN_MEM_POOL : YES
                CLDNN_NV12_TWO_INPUTS : NO
                CLDNN_PLUGIN_PRIORITY : 0
                CLDNN_PLUGIN_THROTTLE : 0
                CLDNN_SOURCES_DUMPS_DIR : ""
                CONFIG_FILE : ""
                DEVICE_ID : ""
                DUMP_KERNELS : NO
                DYN_BATCH_ENABLED : NO
                EXCLUSIVE_ASYNC_REQUESTS : NO
                GPU_THROUGHPUT_STREAMS : 1
                PERF_COUNT : NO
                TUNING_FILE : ""
                TUNING_MODE : TUNING_DISABLED
```

## hello_classification

```
> unzip alexnet.zip.001
> hello_classification.exe alexnet.xml banana.jpg CPU

Top 10 results:

Image banana.jpg

classid probability
------- -----------
954     0.9986986
666     0.0005437
951     0.0002908
114     0.0000703
940     0.0000553
502     0.0000525
659     0.0000387
435     0.0000228
950     0.0000203
910     0.0000185
```