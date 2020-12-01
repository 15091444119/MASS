
#python train_combiner.py \
#export NGPU=2
export CUDA_VISIBLE_DEVICES="3"
python train_combiner.py \
python -m torch.distributed.launch --nproc_per_node=$NGPU --master_addr 192.168.1.1 --master_port 1234 train_combiner.py \
	--exp_name multi_gpu_test                             \
	--data_path /home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain \
	--lgs 'zh-en'                                        \
	--mt_steps "zh-en" \
	--emb_dim 1024                                       \
	--n_layers 6                                         \
	--n_heads 8                                          \
	--dropout 0.1                                        \
	--attention_dropout 0.1                              \
	--gelu_activation true                               \
	--tokens_per_batch 1000 \
	--batch_size 32 \
	--optimizer adam,lr=0.0001 \
	--epoch_size 200000                                  \
	--max_epoch 200                                      \
  --debug_train False \
  --eval_only False \
  --eval_bleu True \
  --group_by_size True
