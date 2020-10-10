export CUDA_VISIBLE_DEVICES="2"
MODEL=/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-wwm-reload/j03q6ubj61/periodic-50.pth
#/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-ft-jointbpe-jointvocab/q6vn71z093/checkpoint.pth
#/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-checkpoint-pretrain/x363q5pus9/periodic-150.pth

python train_combiner.py \
	--exp_name debug                             \
	--data_path ./combiner_data/debug_data \
	--lgs 'zh-en'                                        \
	--encoder_only False                                 \
	--reload_model "$MODEL,$MODEL" \
	--emb_dim 1024                                       \
	--n_layers 6                                         \
	--n_heads 8                                          \
	--dropout 0.1                                        \
	--attention_dropout 0.1                              \
	--gelu_activation true                               \
	--batch_size 100 \
	--optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
	--epoch_size 5000                                  \
	--max_epoch 200                                      \
	--src_bped_words_path ./word_vocab/zh.vocab.bpe \
	--tgt_bped_words_path ./word_vocab/en.vocab.bpe \
	--dict_src_lang zh \
	--dict_tgt_lang en \
	--dict_path /home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/BLI/zh-en/zh-en.txt.sim \
	--codes_path /home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain/codes \
	--combiner_steps "zh,en" \
	--combiner_loss "COS" \
	--combiner "gru" \
  --share_combiner False \
  --n_combiner_layers 4 \
  --validation_metrics _valid-average-loss \
  --bli_preprocess_method 'u' \
  --splitter "CHAR" \
  --origin_context_extractor "before_eos" \
  --combiner_context_extractor "last_time"
#  --reload_encoder_combiner_path /home/data_ti5_d/zhouzh/low-resource-mt/subword-combiner/MASS-unsupNMT/dumped/separate_combiner_gru_cos_4_layer/xt9davossf/best-valid-average-loss.pth \
#  --eval_only True
