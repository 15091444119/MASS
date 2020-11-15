export CUDA_VISIBLE_DEVICES="0"
#MODEL=/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-ft-jointbpe-jointvocab/q6vn71z093/checkpoint.pth
#MODEL=/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-wwm-reload/j03q6ubj61/periodic-50.pth
#/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-wwm-reload/j03q6ubj61/periodic-50.pth
#/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-checkpoint-pretrain/x363q5pus9/periodic-150.pth
MODEL=/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-pretrain-jointbpe-jointvocab/1jnrqg51bz/periodic-100.pth

train(){
python train_combiner.py \
	--exp_name tmp                             \
	--data_path /home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain \
	--lgs 'zh-en'                                        \
	--encoder_only False                                 \
	--reload_model "" \
	--emb_dim 1024                                       \
	--n_layers 6                                         \
	--n_heads 8                                          \
	--dropout 0.1                                        \
	--attention_dropout 0.1                              \
	--gelu_activation true                               \
	--batch_size 64 \
	--optimizer adam,lr=0.0001 \
	--epoch_size 50000                                  \
	--max_epoch 1000                                      \
	--src_lang "zh" \
	--tgt_lang "en" \
	--codes_path /home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain/codes \
	--combiner_loss "COS" \
  --n_combiner_layers 4 \
  --validation_metrics _valid_zh_loss,valid_zh-en_mt_bleu \
  --stopping_criterion valid_zh-en_mt_bleu,10 \
  --splitter ROB \
  --debug_train True \
  --eval_only False \
  --re_encode_rate 0.3 \
  --word_mask_keep_rand 1.0,0.0,0.0 \
  --word_mass 0.5
}
train