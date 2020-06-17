export NGPU=4 
export CUDA_VISIBLE_DEVICES="4,5,6,7"
python -m torch.distributed.launch --nproc_per_node=$NGPU train.py \
	--exp_name ne-en-500w-pretrain-jointbpe-jointvocab                              \
	--data_path /home/data_ti4_c/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/en-ne-joint-bpe-joint-vocab \
	--lgs 'ne-en'                                        \
	--mass_steps 'ne,en'                                 \
	--encoder_only false                                 \
	--emb_dim 1024                                       \
	--n_layers 6                                         \
	--n_heads 8                                          \
	--dropout 0.1                                        \
	--attention_dropout 0.1                              \
	--gelu_activation true                               \
	--tokens_per_batch 2000                              \
	--optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
	--epoch_size 200000                                  \
	--max_epoch 1000                                      \
	--eval_bleu true                                     \
	--word_mass 0.5                                      \
	--min_len 5
