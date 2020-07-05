import sys
import argparse

def parse_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_train")
    parser.add_argument("--src_valid")
    parser.add_argument("--tgt_train")
    parser.add_argument("--tgt_valid")
    params = parser.parse_args()
    return params

def main(params):
    """ cat parallel sentences using seperator "|||", this is the input format of fast align """

    # cat training set
    with open(params.src_train, 'r') as f_src, open(params.tgt_train, 'r') as f_tgt:
        for src_line, tgt_line in zip(f_src, f_tgt):

            # escape empty sentence, which will cause bug in fastalign
            src_line = src_line.rstrip()
            tgt_line = tgt_line.rstrip()

            if src_line == "" or tgt_line == "":
                continue

            print("{} ||| {}".format(src_line, tgt_line))
    
    # cat valid set
    with open(params.src_valid, 'r') as f_src, open(params.tgt_valid, 'r') as f_tgt:
        for src_line, tgt_line in zip(f_src, f_tgt):

            src_line = src_line.rstrip()
            tgt_line = tgt_line.rstrip()
            
            # valid set should not have empty sentence
            assert src_line != "" and tgt_line != ""

            print("{} ||| {}".format(src_line, tgt_line))

if __name__ == "__main__":
    params = parse_params()
    main(params)
