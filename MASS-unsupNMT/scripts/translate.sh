export CUDA_VISIBLE_DEVICES="3"
Model="/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-ft-jointbpe-jointvocab/q6vn71z093/checkpoint.pth"
InputPath=/home/data_ti5_d/zhouzh/low-resource-mt/eval_context_bli/MASS-unsupNMT/word_vocab/zh.vocab.bpe
OutputPath=./backtranslation.txt
SrcLang="zh"
TgtLang="en"

python3 ./translate.py \
  --exp_name "tmp_translate" \
  --model_path $Model \
  --src_lang $SrcLang \
  --tgt_lang $TgtLang \
  --batch_size 64 \
  --output_path $OutputPath  < $InputPath

sed -i -r "s/(@@ )|(@@ ?$)//g" $OutputPath

