export CUDA_VISIBLE_DEVICES="0"
FantiFinetuned="/home/data_ti4_c/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/en-zh-500w-ft-jointbpe-jointvocab/31b14y3hsf/59.pth" 
FantiPretrained="/home/data_ti4_c/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/en-zh-500w-pretrain-jointbpe-jointvocab/9pvn3bwu2s/checkpoint.pth" 
JiantiSupervised="/home/data_ti4_c/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-supervised/u7s91zmjk2/checkpoint.pth"
JiantiPretrained="/home/data_ti4_c/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-pretrain-jointbpe-jointvocab/1jnrqg51bz/checkpoint.pth"
JiantiFinetuned="/home/data_ti4_c/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-ft-jointbpe-jointvocab/q6vn71z093/checkpoint.pth"
python3 -m src.evaluation.eval_attention.eval_attention \
	--exp_name "eval_attention" \
	--model_path $JiantiPretrained \
	--src_lang "zh" \
	--tgt_lang "en" \
