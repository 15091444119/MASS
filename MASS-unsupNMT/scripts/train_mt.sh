#MODEL=/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-pretrain-jointbpe-jointvocab/1jnrqg51bz/periodic-100.pth
#export NGPU=2
#python train_combiner.py \
#python -m torch.distributed.launch --nproc_per_node=$NGPU train_combiner.py \
export CUDA_VISIBLE_DEVICES="1"
python train_combiner.py \
	--exp_name test_mass                             \
	--data_path /home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain \
	--lgs 'zh-en'                                        \
	--mass_steps 'zh,en' \
	--emb_dim 1024                                       \
	--n_layers 2                                         \
	--word_mass 0.5 \
	--n_heads 8                                          \
	--dropout 0.1                                        \
	--attention_dropout 0.1                              \
	--gelu_activation true                               \
	--batch_size 32 \
	--optimizer adam,lr=0.0001 \
	--epoch_size 100000                                  \
	--max_epoch 200                                      \
  --debug_train False \
  --eval_only False \
  --eval_bleu True \
  --group_by_size False \
  --eval_mass_steps "zh,en"
