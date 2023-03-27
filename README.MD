# Riva for ASR

Deploy a Custom Acoustic Model (Conformer-CTC)

Riva[简介](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/overview.html), [github](https://github.com/nvidia-riva), [tutorial](https://github.com/nvidia-riva/tutorials/blob/r2.9.0/asr-deploy-conformer-ctc.ipynb)

tensorrt --> triton --> riva

# install

从ngc下载镜像方法, 参考[博客](https://blog.csdn.net/Deaohst/article/details/128789824), ngc使用[参考](https://blog.csdn.net/weixin_44831720/article/details/117956175)

requirements:

1. [注册](https://ngc.nvidia.com/account)nvidia-ngc账户
2. ngc客户端[安装](https://ngc.nvidia.com/setup/installers/cli), 选择对应的系统，win/AMDlinux/ARMlinux/MacOS
3. ngc配置 `ngc config set`
   + 从ngc账户中生成 `API Key`和用户密码
4. 终端中登录nvidia-io,`docker login nvcr.io`
5. [quick_start](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/quick-start-guide.html)


# [deploy](https://github.com/nvidia-riva/tutorials/blob/main/asr-deploy-conformer-ctc.ipynb):

1. ``ngc registry resource download-version nvidia/riva/riva_quickstart:2.7.0``

2. 根据实际配置修改`config.sh`，

    `cd riva_quickstart:2.7.0 && chmod +x riva_init.sh && bash riva_init.sh`

3. conformer模型[下载](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/speechtotext_zh_cn_conformer/files)

    >conformer普通话语音识别模型[介绍](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/speechtotext_zh_cn_conformer)

4. `riva`转`rmir`:
    ```bash
    docker run --rm --gpus 0 -v /home/vastai/workspace/Riva/riva_quickstart_v2.7.0/riva-model-repo:/data nvcr.io/nvidia/riva/riva-speech:2.7.0-servicemaker -- \
        riva-build speech_recognition \
            /data/rmir/Conformer-CTC-L-char-zh-CN-Riva-ASR-set-3.0-avg.rmir \
            /data/Conformer-CTC-L-char-zh-CN-Riva-ASR-set-3.0-avg.riva \
            --offline \
            --name=asr_offline_conformer_ctc_pipeline \
            --decoder_type=greedy \
            --ms_per_timestep=40 \
            --chunk_size=4.8 \
            --left_padding_size=1.6 \
            --right_padding_size=1.6 \
            --max_batch_size=16 \
            --nn.fp16_needs_obey_precision_pass \
            --featurizer.use_utterance_norm_params=False \
            --featurizer.precalc_norm_time_steps=0 \
            --featurizer.precalc_norm_params=False \
            --featurizer.max_batch_size=512 \
            --featurizer.max_execution_batch_size=512 \
            --language_code=zh-CN
    ```        

5. rmri 序列化：
    ```bash
    docker run --rm --gpus 0 -v /home/vastai/workspace/Riva/riva_quickstart_v2.7.0/riva-model-repo:/data nvcr.io/nvidia/riva/riva-speech:2.7.0-servicemaker -- \
        riva-deploy -f  \
            /data/rmir/Conformer-CTC-L-char-zh-CN-Riva-ASR-set-3.0-avg.rmir \
            /data/models/

    ```
    > 生成onnx[模型](http://10.23.4.220:8080/trt/customer/comformer/RIVA%20Conformer%20ASR%20Mandarin/onnx/)和trt engine文件和riva部署模型[文件](http://10.23.4.220:8080/trt/customer/comformer/RIVA%20Conformer%20ASR%20Mandarin/trt_engine/)

6. 启动服务 

    > `bash riva_start.sh`

7. 客户端`python_client`请求
    ```bash
    cd python-clients

    python scripts/asr/transcribe_file_offline.py --input-file ../riva_quickstart_v2.7.0/data/BAC009S0724W0121.wav --simulate-realtime --show-intermediate --language-code zh-CN

    # results {
    #   alternatives {
    #       transcript: "广州市房地产中介协会分析 "
    #       confidence: -0.000287532806
    #   }
    #   channel_tag: 1
    #   audio_processed: 4.8
    # }
    # Final transcript: 广州市房地产中介协会分析 
    # pipeline end, time: 0.0661 s
    ```
    [transcribe_file_offline.py](./python-clients/scripts/asr/transcribe_file_offline.py)

# dependency

+ 推理服务[Triton_inference_service](https://github.com/triton-inference-server/server)

+ 推理框架[TensorRt](https://github.com/NVIDIA/TensorRT)

+ 训练框架[Nemo](https://github.com/nvidia/nemo)