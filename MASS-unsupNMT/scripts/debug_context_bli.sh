export CUDA_VISIBLE_DEVICES="3"
python3 -m src.evaluation.eval_context_bli.eval \
    --model_path /home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-pretrain-jointbpe-jointvocab/1jnrqg51bz/periodic-100.pth \
    --src_bped_words_path ./debug_src_words.txt \
    --tgt_bped_words_path ./debug_tgt_words.txt \
    --src_lang zh \
    --tgt_lang en \
    --dict_path ./debug_dict.txt
