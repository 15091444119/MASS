export CUDA_VISIBLE_DEVICES=3
MODEL=/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-pretrain-jointbpe-jointvocab/1jnrqg51bz/periodic-100.pth
python3 -m src.evaluation.eval_context_combiner.dataset \
  --checkpoint ./dumped/combiner_cos_1.0/j35ib7fmxs/checkpoint.pth \
  --bped_src /home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain/valid.en-zh.zh \
  --bped_tgt /home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain/valid.en-zh.en  \
  --src_lang zh \
  --tgt_lang en \
  --alignments /home/data_ti5_d/zhouzh/low-resource-mt/alignment_data/zh-en/final_align.intersect.valid \
  --batch_size 5
 # --mass_checkpoint_for_hack $MODEL \
 # --average_hack