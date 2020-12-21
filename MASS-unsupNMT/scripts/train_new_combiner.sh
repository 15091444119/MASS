Lang=zh
Codes=/home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain/codes
DataPrefix=/home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain/mass_context_combiner_data/zh
Mass=/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-checkpoint-pretrain/x363q5pus9/periodic-200.pth


export CUDA_VISIBLE_DEVICES="1"
python3 train_new_context_combiner.py \
  --exp_name "test" \
  --batch_size 32 \
  --epoch 10000 \
  --max_epoch 200 \
  --reload_model $Mass \
  --combiner "last_token" \
  --lang $Lang \
  --emb_dim 1024 \
  --sinusoidal_embeddings False \
  --n_head 8 \
  --n_layer 4 \
  --splitter "ROB" \
  --combiner_loss "COS" \
  --codes_path $Codes \
  --combiner_train_data $DataPrefix.dev.txt \
  --combiner_dev_data $DataPrefix.dev.txt \
  --combiner_test_data $DataPrefix.test.txt