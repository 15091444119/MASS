python3 -m src.evaluation.eval_context_combiner.dataset \
  --checkpoint \
  --bped_src /home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain/valid.en-zh.zh \
  --bped_tgt /home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain/valid.en-zh.en  \
  --src_lang zh \
  --tgt_lang en \
  --alignments  \
  --batch_size 32