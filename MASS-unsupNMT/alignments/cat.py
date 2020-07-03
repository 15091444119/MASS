import sys

def main():
    """ cat parallel sentences using seperator "|||", this is the input format of fast align """
    src = sys.argv[1]
    tgt = sys.argv[2]
    with open(src, 'r') as f_src, open(tgt, 'r') as f_tgt:
        for src_line, tgt_line in zip(f_src, f_tgt):
            print("{} ||| {}".format(src_line.rstrip(), tgt_line.rstrip()))

if __name__ == "__main__":
    main()
