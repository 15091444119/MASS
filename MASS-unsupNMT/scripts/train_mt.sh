MODEL=/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-pretrain-jointbpe-jointvocab/1jnrqg51bz/periodic-100.pth
#export NGPU=2
#python train_combiner.py \
#python -m torch.distributed.launch --nproc_per_node=$NGPU train_combiner.py \
export CUDA_VISIBLE_DEVICES="1"
python train_combiner.py \
	--exp_name test_mass                             \
	--encoder_type combiner \
	--data_path /home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain \
	--lgs 'zh-en'                                        \
	--reload_model "$MODEL,$MODEL" \
	--emb_dim 1024                                       \
	--n_layers 6                                         \
	--n_heads 8                                          \
	--dropout 0.1                                        \
	--attention_dropout 0.1                              \
	--gelu_activation true                               \
	--optimizer adam,lr=0.0001 \
  --combiner "last_token" \
  --splitter "ROB" \
	--codes_path /home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain/codes \
  --combiner_loss "COS" \
  --re_encode_rate 0.3 \
	--batch_size 64 \
	--epoch_size 200000                                  \
	--max_epoch 200                                      \
  --group_by_size False \
	--mass_steps 'zh,en' \
	--explicit_mass_steps 'zh,en' \
	--word_mass 0.5 \
  --eval_mass_steps "zh,en" \
  --eval_explicit_mass_steps "zh,en" \
  --eval_mt_steps "zh-en,en-zh" \
  --eval_only False \
  --eval_bleu True \
  --debug_train False
