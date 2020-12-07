export CUDA_VISIBLE_DEVICES=2
MODEL=/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-pretrain-jointbpe-jointvocab/1jnrqg51bz/periodic-100.pth
COS_03=./dumped/combiner_cos_0.3/xgznxn7f9x/checkpoint.pth
COS_10=./dumped/combiner_cos_1.0/j35ib7fmxs/checkpoint.pth
MSE_03=/home/data_ti5_d/zhouzh/low-resource-mt/context-combiner/MASS-unsupNMT/dumped/combiner_mes_0.3/fzn3iwxr68/checkpoint.pth
python3 -m src.evaluation.eval_context_combiner.__init__ \
  --checkpoint ${MSE_03} \
  --bped_src /home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain/valid.en-zh.zh \
  --bped_tgt /home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain/valid.en-zh.en  \
  --src_lang zh \
  --tgt_lang en \
  --alignments /home/data_ti5_d/zhouzh/low-resource-mt/alignment_data/zh-en/final_align.intersect.valid \
  --batch_size 32 \
  --mass_checkpoint_for_hack $MODEL \
  --average_hack