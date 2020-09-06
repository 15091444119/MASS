export CUDA_VISIBLE_DEVICES="3"
python3 -m src.evaluation.eval_context_bli.eval \
    --model_path /home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-pretrain-jointbpe-jointvocab/1jnrqg51bz/periodic-100.pth \
    --src_bped_words_path /home/user_data55/zhouzh/dictionary/new_data/embeddings/zh.vocab.bpe \
    --tgt_bped_words_path  /home/user_data55/zhouzh/dictionary/new_data/embeddings/en.vocab.bpe \
    --src_lang zh \
    --tgt_lang en \
    --dict_path /home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/BLI/zh-en/zh-en.txt 
