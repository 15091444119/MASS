export NGPU=4 
export CUDA_VISIBLE_DEVICES="4,5,6,7"
python -m torch.distributed.launch --nproc_per_node=$NGPU train.py \
	--exp_name ne-en-500w-pretrain-jointbpe-jointvocab                              \
	--data_path /home/data_ti4_c/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/en-ne-joint-bpe-joint-vocab \
	--lgs 'zh-en'                                        \
	--encoder_only True                                \
	--emb_dim 1024                                       \
	--n_layers 6                                         \
	--n_heads 8                                          \
	--dropout 0.1                                        \
	--attention_dropout 0.1                              \
	--gelu_activation true                               \
	--batch_size 128 \
	--optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
	--epoch_size 128                                  \
	--max_epoch 5                                      \
