export CUDA_VISIBLE_DEVICES="0"
python3 -m src.evaluation.eval_encoder_token2word.eval_encoder_token2word \
	--reloaded "/home/data_ti4_c/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-pretrain-jointbpe-jointvocab/1jnrqg51bz/checkpoint.pth" \
        --exp_name eval_token2word
