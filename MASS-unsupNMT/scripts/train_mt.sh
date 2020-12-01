
export NGPU=3
#python train_combiner.py \
export CUDA_VISIBLE_DEVICES="5,6,7"
python -m torch.distributed.launch --nproc_per_node=$NGPU train_combiner.py \
	--exp_name tmp_single                             \
	--data_path /home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain \
	--lgs 'zh-en'                                        \
	--mt_steps "zh-en" \
	--emb_dim 1024                                       \
	--n_layers 6                                         \
	--n_heads 8                                          \
	--dropout 0.1                                        \
	--attention_dropout 0.1                              \
	--gelu_activation true                               \
	--tokens_per_batch 2000 \
	--batch_size 32 \
	--optimizer adam,lr=0.0001 \
	--epoch_size 200000                                  \
	--max_epoch 200                                      \
  --debug_train False \
  --eval_only False \
  --eval_bleu True
