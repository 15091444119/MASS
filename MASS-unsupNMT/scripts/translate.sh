Model=/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-pretrain-jointbpe-jointvocab/1jnrqg51bz/periodic-100.pth
InputPath=/home/data_ti5_d/zhouzh/low-resource-mt/eval_context_bli/MASS-unsupNMT/word_vocab/zh.vocab.bpe
OutputPath=./output.txt
SrcLang="zh"
TgtLang="en"

python3 ./translate.py \
  --model_path $Model \
  --src_lang $SrcLang \
  --tgt_lang $TgtLang \
  --output_path $OutputPath \ < $InputPath

sed -i -r "s/(@@ )|(@@ ?$)//g" $OutputPath

