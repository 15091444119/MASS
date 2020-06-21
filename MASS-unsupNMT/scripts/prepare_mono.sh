set -e
set -u
Wiki="/home/zhouzh/data/raw_data/mono/downloaded_raw_wiki/bz2/enwiki-latest-pages-articles.xml.bz2"
lg=encn # we convert traditional chinese to simplified chinese so we add cn to the language prefix
toklg=en
DestDir="/home/zhouzh/data/tokenized_data/mono"
ToolsPath="/home/data_ti4_c/zhouzh/low-resource-mt/tools"
Tokenize=$ToolsPath/tokenize.sh
LowerRemoveAccent=$ToolsPath/lowercase_and_remove_accent.py
T2S="python3 $ToolsPath/convert_t2s.py"
SplitSentence="python3 $ToolsPath/split_sentence.py"

if [ ! -f $DestDir/$lg.all ]; then
  python3 $ToolsPath/wikiextractor/WikiExtractor.py $Wiki --processes 8 -q -o - \
  | sed "/^\s*\$/d" \
  | grep -v "^<doc id=" \
  | grep -v "</doc>\$" \
  | $T2S \
  | $Tokenize $toklg \
  | python3 $LowerRemoveAccent \
  > $DestDir/$lg.all
fi

echo "*** Split into train / valid / test ***"
split_data() {
    get_seeded_random() {
        seed="$1"; openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null
    };
    NLINES=`wc -l $1  | awk -F " " '{print $1}'`;
    NTRAIN=$((NLINES - 10000));
    NVAL=$((NTRAIN + 5000));
    shuf --random-source=<(get_seeded_random 42) $1 | head -$NTRAIN             > $2;
    shuf --random-source=<(get_seeded_random 42) $1 | head -$NVAL | tail -5000  > $3;
    shuf --random-source=<(get_seeded_random 42) $1 | tail -5000                > $4;
}
split_data $DestDir/$lg.all $DestDir/$lg.train $DestDir/$lg.valid $DestDir/$lg.test

# for back translation training, we split sentences because we don't use stream dataset
for splt in "train" "valid" "test"; do
    $SplitSentence $DestDir/$lg.$splt $DestDir/$lg.$splt.split_sentence
done
