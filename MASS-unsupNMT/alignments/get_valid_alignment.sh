"""
valid alignment will be saved in expdir/final_align.valid
"""
set -e
set -u

DataPath="/home/zhouzh/data/tokenized_data/para/zh-en"
FastAlign="/home/data_ti4_c/zhouzh/low-resource-mt/tools/fast_align/build"
SrcTrain=$DataPath/train.en-zh.zh 
TgtTrain=$DataPath/train.en-zh.en 
SrcValid=$DataPath/valid.en-zh.zh 
TgtValid=$DataPath/valid.en-zh.en 

ExpDir="./cn-zh-en"
if [ -e $ExpDir ]; then
    rm -r $ExpDir
fi

mkdir -p $ExpDir
SrcAll=$ExpDir/zh
TgtAll=$ExpDir/en
AlignInput=$ExpDir/all
ValidNum=$(cat $SrcValid | wc -l)
Forward=$ExpDir/forward.align
Backward=$ExpDir/backward.align
FinalAlign=$ExpDir/final_align.all
FinalValidAlign=$ExpDir/final_align.valid

# cat data
echo $ValidNum
cat $SrcTrain $SrcValid > $SrcAll
cat $TgtTrain $TgtValid > $TgtAll
python3 ./cat.py $SrcAll $TgtAll > $AlignInput

# fastalign
$FastAlign/fast_align -i $AlignInput -d -o -v > $Forward
$FastAlign/fast_align -i $AlignInput -d -o -v -r > $Backward
$FastAlign/atools -i $Forward -j $Backward -c grow-diag-final-and > $FinalAlign

tail -$ValidNum $FinalAlign > $FinalValidAlign
