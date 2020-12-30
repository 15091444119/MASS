export CUDA_VISIBLE_DEVICES="3"
FTMODEL=/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-ft-jointbpe-jointvocab/q6vn71z093/checkpoint.pth
FTTRAINVOCAB=/home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain/zh.count.txt
SUPMODEL=/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-supervised/u7s91zmjk2/checkpoint.pth
SUPTRAINVOCAB=/home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain/en-zh.zh.count.txt
PRETRAIN=/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-pretrain-jointbpe-jointvocab/1jnrqg51bz/periodic-100.pth
ZHEN_ZH=/home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/fr_en_split_500w/valid.en-fr.fr
ZHEN_CODES=/home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain/codes

FRENVOCAB=/home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/fr_en_split_500w/vocab
FRENFTMODEL=/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/en-fr-500w-split-ft/qmb5bz8o3v/checkpoint.pth
FRENPREMODEL=/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/en-fr-500w-split-pretrain-goon/9gd15xfdjn/checkpoint.pth
FREN_FR=/home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/fr_en_split_500w/valid.en-fr.fr
FREN_CODES=/home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/fr_en_split_500w/codes

python3 -m src.evaluation.eval_bpe_encoding \
    --model_path $FRENFTMODEL \
    --lang fr \
    --codes_path ${FREN_CODES} \
    --train_vocab $FRENVOCAB \
    --batch_size 32 \
    --corpus_path ${FREN_FR}