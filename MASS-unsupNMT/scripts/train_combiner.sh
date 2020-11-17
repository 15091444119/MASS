#MODEL=/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-ft-jointbpe-jointvocab/q6vn71z093/checkpoint.pth
#MODEL=/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-wwm-reload/j03q6ubj61/periodic-50.pth
#/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-wwm-reload/j03q6ubj61/periodic-50.pth
#/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-checkpoint-pretrain/x363q5pus9/periodic-150.pth
MODEL=/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-pretrain-jointbpe-jointvocab/1jnrqg51bz/periodic-100.pth
train(){
#export NGPU=2
#export CUDA_VISIBLE_DEVICES="1,3"
#python -m torch.distributed.launch --nproc_per_node=$NGPU train_combiner.py \
export CUDA_VISIBLE_DEVICES="0"
python train_combiner.py \
	--exp_name tmp_all                             \
	--data_path /home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain \
	--lgs 'zh-en'                                        \
	--encoder_only False                                 \
	--reload_model "$MODEL,$MODEL" \
	--emb_dim 1024                                       \
	--n_layers 6                                         \
	--n_heads 8                                          \
	--dropout 0.1                                        \
	--attention_dropout 0.1                              \
	--gelu_activation true                               \
	--batch_size 64 \
	--optimizer adam,lr=0.0001 \
	--epoch_size 100000                                  \
	--max_epoch 200                                      \
	--src_lang "zh" \
	--tgt_lang "en" \
	--codes_path /home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain/codes \
	--combiner_loss "COS" \
  --n_combiner_layers 2 \
  --splitter ROB \
  --debug_train False \
  --eval_only False \
  --re_encode_rate 0.3 \
  --word_mask_keep_rand 1.0,0.0,0.0 \
  --word_mass 0.5 \
  --eval_bleu True
}
train