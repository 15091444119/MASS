export CUDA_VISIBLE_DEVICES="3"
a(){
python3 -m src.evaluation.eval_context_bli.eval \
    --model_path /home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-wwm-reload/j03q6ubj61/periodic-50.pth \
    --src_bped_words_path ./word_vocab/zh.vocab.bpe \
    --tgt_bped_words_path ./word_vocab/en.vocab.bpe \
    --src_lang zh \
    --tgt_lang en \
    --dict_path /home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/BLI/zh-en/zh-en.txt.sim \
    --context_extractor average
}

c(){
python3 -m src.evaluation.eval_context_bli.eval \
    --model_path /home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-wwm-reload/o82ssovc0h/periodic-50.pth \
    --src_bped_words_path ./word_vocab/zh.vocab.bpe \
    --tgt_bped_words_path ./word_vocab/en.vocab.bpe \
    --src_lang zh \
    --tgt_lang en \
    --dict_path /home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/BLI/zh-en/zh-en.txt.sim \
    --context_extractor average
}
b(){
python3 -m src.evaluation.eval_context_bli.eval \
    --model_path /home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/en-zh-500w-ft-jointbpe-jointvocab/31b14y3hsf/checkpoint.pth \
    --src_bped_words_path ./word_vocab/zh.vocab.bpe \
    --tgt_bped_words_path ./word_vocab/en.vocab.bpe \
    --src_lang zh \
    --tgt_lang en \
    --dict_path /home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/BLI/zh-en/zh-en.txt.sim \
    --context_extractor average
}

a
b
c
