DataPrefix=/home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain/mass_emb_combiner_data/zh
MODEL=/home/data_ti5_d/zhouzh/low-resource-mt/MASS/MASS-unsupNMT/dumped/cn-en-zh-500w-pretrain-jointbpe-jointvocab/1jnrqg51bz/periodic-100.pth

export CUDA_VISIBLE_DEVICES="5"
python3 train_emb_combiner.py \
    --exp_name test \
    --eval_only True \
    --eval_loss_sentences -1 \
    --max_epoch 200 \
    --epoch_size 30000 \
    --optimizer adam,lr=0.0001 \
    --mass_model $MODEL \
    --train $DataPrefix.train.txt \
    --dev $DataPrefix.dev.txt \
    --test $DataPrefix.test.txt \
    --batch_size 64 \
    --splitter "ROB" \
    --codes_path /home/data_ti5_d/zhouzh/low-resource-mt/XLM_MASS_preprocessed_data/pretrain/cn-split-sen-zh-en-pretrain/codes \
    --combiner_type "average" \
    --context_extractor_type average \
    --loss "COS"