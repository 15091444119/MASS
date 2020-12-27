export CUDA_VISIBLE_DEVICES=1
SrcModel=/home/data_ti5_d/zhouzh/low-resource-mt/combiner/MASS-unsupNMT/dumped/last_token_combine_label_embedding_zh/dvjmx4ovmt/best-dev-combiner-word-average-loss.pth
TgtModel=/home/data_ti5_d/zhouzh/low-resource-mt/combiner/MASS-unsupNMT/dumped/last_token_combine_label_embedding_en/bgocsx8krq/best-dev-combiner-word-average-loss.pth
MassModel=/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-checkpoint-pretrain/x363q5pus9/periodic-200.pth

python3 -m src.evaluation.eval_context_combiner.eval_one_word_combiner \
  --src_combiner ${SrcModel} \
  --tgt_combiner ${TgtModel} \
  --mass_model ${MassModel} \
  --bped_src /home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain/valid.en-zh.zh \
  --bped_tgt /home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain/valid.en-zh.en  \
  --src_lang zh \
  --tgt_lang en \
  --alignments /home/data_ti5_d/zhouzh/low-resource-mt/alignment_data/data/zh-en/final_align.intersect.valid \
  --batch_size 32 \
  --metric "COS"
  #--mass_checkpoint_for_hack $MODEL \
  #--average_hack
