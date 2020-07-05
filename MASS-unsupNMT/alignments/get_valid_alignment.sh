# valid alignment will be saved in expdir/final_align.valid

set -e
set -u

DataPath="/home/zhouzh/data/tokenized_data/para/zh-en"
FastAlign="/home/data_ti4_c/zhouzh/low-resource-mt/tools/fast_align/build"
SrcTrain=$DataPath/train.en-zh.zh 
TgtTrain=$DataPath/train.en-zh.en 
SrcValid=$DataPath/valid.en-zh.zh 
TgtValid=$DataPath/valid.en-zh.en 

ExpDir="./zh-en"
if [ -e $ExpDir ]; then
    rm -r $ExpDir
fi

mkdir -p $ExpDir
AlignInput=$ExpDir/all
Forward=$ExpDir/forward.align
Backward=$ExpDir/backward.align
FinalAlignGdfa=$ExpDir/final_align.gdfa.all
FinalAlignIntersect=$ExpDir/final_align.intersect.all
FinalValidAlignGdfa=$ExpDir/final_align.gdfa.valid
FinalValidAlignIntersect=$ExpDir/final_align.intersect.valid

# cat data
ValidNum=$(cat $SrcValid | wc -l)
python3 ./generate_fast_align_input.py --src_train $SrcTrain --src_valid $SrcValid --tgt_train $TgtTrain --tgt_valid $TgtValid > $AlignInput

#Debug=$ExpDir/debug
#head -500000 $AlignInput > $Debug

# fastalign
$FastAlign/fast_align -i $AlignInput -d -o -v > $Forward
$FastAlign/fast_align -i $AlignInput -d -o -v -r > $Backward
$FastAlign/atools -i $Forward -j $Backward -c grow-diag-final-and > $FinalAlignGdfa
$FastAlign/atools -i $Forward -j $Backward -c intersect > $FinalAlignIntersect

tail -$ValidNum $FinalAlignGdfa > $FinalValidAlignGdfa
tail -$ValidNum $FinalAlignIntersect > $FinalValidAlignIntersect
