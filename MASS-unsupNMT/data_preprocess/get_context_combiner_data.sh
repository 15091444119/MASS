for Lang in "zh" "en"; do
  for Part in "dev" "test"; do
    python3 ./get_context_combiner_data.py \
      --vocab /home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain/mass_emb_combiner_data/$Lang.$Part.txt \
      --data /home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain/train.$Lang \
      --output /home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain/mass_context_combiner_data/$Lang.$Part.txt
  done
done
