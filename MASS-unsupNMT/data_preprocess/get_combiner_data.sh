DestDir=/home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain/mass_emb_combiner_data
Vocab=/home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain/vocab
SrcPath=/home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain/train.zh
TgtPath=/home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain/train.en
Src=zh
Tgt=en


python3 ./get_combiner_data.py \
  --src_lang $Src \
  --tgt_lang $Tgt \
  --vocab $Vocab \
  --train_src $SrcPath \
  --train_tgt $TgtPath \
  --dest_dir $DestDir