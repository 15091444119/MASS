export CUDA_VISIBLE_DEVICES="1"
Model="/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-supervised/u7s91zmjk2/checkpoint.pth"
LongModel="/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-wwm-reload/j03q6ubj61/periodic-50.pth"
FtModel="/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-ft-jointbpe-jointvocab/q6vn71z093/checkpoint.pth"
InputPath="./valid-data/zh-en/zh.txt.bpe"
OutputDir="./model_output/100ft"
OutputWithOutBpe="$OutputDir/zh-en.txt"
OutputPath="$OutputDir/zh-en.txt.bpe"
RefWithOutBpe="./valid-data/zh-en/en.txt"
SrcLang="zh"
TgtLang="en"

rm -r $OutputDir
mkdir -p $OutputDir

python3 ./translate.py \
  --exp_name "tmp_translate" \
  --model_path $FtModel \
  --src_lang $SrcLang \
  --tgt_lang $TgtLang \
  --batch_size 64 \
  --code_path /home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain/codes \
  --output_path $OutputPath  < $InputPath

cp $OutputPath $OutputWithOutBpe
sed -i -r "s/(@@ )|(@@ ?$)//g" $OutputWithOutBpe

./src/evaluation/multi-bleu.perl $RefWithOutBpe  < $OutputWithOutBpe > $OutputPath.bleuscore

cat $OutputPath.bleuscore