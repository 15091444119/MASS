import argparse
import pysbd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()

    seg = pysbd.Segmenter(clean=False)

    with open(args.input, 'r') as f_in, open(args.output, 'w') as f_out:
        for line in f_in:
            sentences = seg.segment(line.rstrip())
            for sentence in sentences:
                f_out.writelines("{}\n".format(sentence))