# Nemo

nvidia 推出的一款ASR\TTS\LLM\NLP模型训练框架。支持较多的[模型](https://github.com/nvidia/nemo#key-features)应用。


+ Speech processing
    + Automatic Speech Recognition (ASR)
        + Supported models: Jasper, QuartzNet, CitriNet, **Conformer-CTC**, Conformer-Transducer, Squeezeformer-CTC, Squeezeformer-Transducer, ContextNet, LSTM-Transducer (RNNT), LSTM-CTC, FastConformer-CTC, FastConformer-Transducer...
        + Supports CTC and Transducer/RNNT losses/decoders
        + NeMo Original Multi-blank Transducers
        + Beam Search decoding
        + Language Modelling for ASR: N-gram LM in fusion with Beam Search decoding, Neural Rescoring with  + Transformer
        + Streaming and Buffered ASR (CTC/Transducer) - Chunked Inference Examples
        + Support of long audios for Conformer with memory efficient local attention

部分模型已提供pre_train模型，拿来即部署。