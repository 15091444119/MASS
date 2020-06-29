export CUDA_VISIBLE_DEVICES="0"
python3 -m src.evaluation.eval_attention.eval_attention \
	--exp_name "eval_attention" \
	--model_path "/home/data_ti4_c/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/en-zh-500w-ft-jointbpe-jointvocab/31b14y3hsf/59.pth" \
	--src_lang "zh" \
	--tgt_lang "en" \
	--src_text "./test.txt"
