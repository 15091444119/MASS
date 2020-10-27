export CUDA_VISIBLE_DEVICES="3"
FTMODEL=/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-ft-jointbpe-jointvocab/q6vn71z093/checkpoint.pth
PRETRAINMODEL=/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-ft-jointbpe-jointvocab/q6vn71z093/checkpoint.pth
FTTRAINVOCAB=/home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain/zh.count.txt
SUPMODEL=/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-supervised/u7s91zmjk2/checkpoint.pth
SUPTRAINVOCAB=/home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain/en-zh.zh.count.txt
PRETRAIN=/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-pretrain-jointbpe-jointvocab/1jnrqg51bz/periodic-100.pth
python3 -m src.evaluation.eval_bpe_encoding \
    --model_path $PRETRAINMODEL \
    --lang zh \
    --codes_path /home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain/codes \
    --train_vocab $FTTRAINVOCAB \
    --batch_size 32 \
    --corpus_path /home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain/valid.en-zh.zh

