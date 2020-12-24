export CUDA_VISIBLE_DEVICES="0"
#Model=/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-ft-jointbpe-jointvocab/q6vn71z093/checkpoint.pth
Model=/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-supervised/u7s91zmjk2/checkpoint.pth
Src=/home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain/valid.en-zh.zh
python3 -m src.evaluation.eval_attention.eval_subword_attention_rate \
	--exp_name "eval_attention" \
	--model_path $Model \
	--src_lang "zh" \
	--tgt_lang "en" \
	--src_path $Src
