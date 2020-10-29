export CUDA_VISIBLE_DEVICES="1"
#MODEL=/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-ft-jointbpe-jointvocab/q6vn71z093/checkpoint.pth
#MODEL=/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-wwm-reload/j03q6ubj61/periodic-50.pth
#/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-wwm-reload/j03q6ubj61/periodic-50.pth
#/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-checkpoint-pretrain/x363q5pus9/periodic-150.pth
MODEL=/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-pretrain-jointbpe-jointvocab/1jnrqg51bz/periodic-100.pth
SPLIT="CHAR"
COMBINER="gru"

train(){
  trained_lang=$1
  other_lang=$2
python train_combiner.py \
	--exp_name eval-pretrain-$SPLIT-$COMBINER-"$trained_lang"                             \
	--data_path ./combiner_data \
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
	--epoch_size 50000                                  \
	--max_epoch 200                                      \
	--bped_words_path ./word_vocab/"$trained_lang".vocab.bpe \
	--trained_lang "$trained_lang" \
	--other_lang "$other_lang" \
	--codes_path /home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain/codes \
	--combiner_steps "$trained_lang" \
	--combiner_loss "COS" \
	--combiner $COMBINER \
  --n_combiner_layers 4 \
  --validation_metrics _valid_loss,valid_whole_combiner_acc_top1_acc \
  --stopping_criterion valid_whole_combiner_acc_top1_acc,20 \
  --bli_preprocess_method 'u' \
  --splitter $SPLIT \
  --origin_context_extractor "before_eos" \
  --combiner_context_extractor "average" \
  --reload_combiner_path ./dumped/pretrain100-CHAR-gru-zh/dvptir0l0h/best-valid_whole_combiner_acc_top1_acc.pth \
  --eval_only True

}

train zh en
#train en zh