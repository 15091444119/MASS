Lang=zh
Codes=/home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain/codes
DataPrefix=/home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain/mass_context_combiner_data/$Lang
Raw=/home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain/valid.$Lang
Mass=/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-checkpoint-pretrain/x363q5pus9/periodic-200.pth
ReloadCombiner=/home/data_ti5_d/zhouzh/low-resource-mt/combiner/MASS-unsupNMT/dumped/last_token_combine_label_embedding_zh/dvjmx4ovmt/best-dev-combiner-word-average-loss.pth

export CUDA_VISIBLE_DEVICES="1"

word_input(){
python3 train_new_context_combiner.py \
  --exp_name "word_input_$Lang" \
  --batch_size 64 \
  --epoch_size 50000 \
  --max_epoch 200 \
  --reload_model $Mass \
  --combiner "word_input" \
  --lang $Lang \
  --emb_dim 1024 \
  --sinusoidal_embeddings False \
  --n_head 8 \
  --n_another_context_encoder_layer 2 \
  --n_word_combiner_layer 2 \
  --splitter "ROB" \
  --combiner_loss "COS" \
  --codes_path $Codes \
  --combiner_train_data $DataPrefix.train.txt \
  --combiner_dev_data $DataPrefix.dev.txt \
  --combiner_test_data $DataPrefix.test.txt \
  --stopping_criterion "_dev-combiner-word-average-loss,20" \
  --validation_metrics "_dev-combiner-word-average-loss"
}

last_token(){
python3 -m src.context_combiner.train_context_combiner \
  --exp_name "eval_$Lang" \
  --batch_size 64 \
  --epoch_size 50000 \
  --max_epoch 200 \
  --reload_model $Mass \
  --combiner "last_token" \
  --combine_label_embedding True \
  --lang $Lang \
  --emb_dim 1024 \
  --sinusoidal_embeddings False \
  --n_head 8 \
  --n_layer 2 \
  --splitter "ROB" \
  --combiner_loss "COS" \
  --codes_path $Codes \
  --train_dataset_type "multi_word_sentence" \
  --combiner_train_data $Raw \
  --combiner_dev_data $DataPrefix.dev.txt \
  --combiner_test_data $DataPrefix.test.txt \
  --stopping_criterion "_dev-combiner-word-average-loss,20" \
  --validation_metrics "_dev-combiner-word-average-loss"
}

eval_average(){
  python3 train_new_context_combiner.py \
  --exp_name "average" \
  --batch_size 64 \
  --eval_only True \
  --epoch_size 50000 \
  --max_epoch 200 \
  --reload_model $Mass \
  --combiner "average" \
  --lang $Lang \
  --splitter "ROB" \
  --combiner_loss "COS" \
  --codes_path $Codes \
  --combiner_train_data $DataPrefix.dev.txt \
  --combiner_dev_data $DataPrefix.dev.txt \
  --combiner_test_data $DataPrefix.test.txt
}

last_token