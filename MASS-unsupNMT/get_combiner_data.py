import sys


def generate_combiner_data(path, output_path):
    """
    Read bped vocab, select word with more than one token and didn't split by bpe to learn the combiner
    Params:
        path: string
            Input bped vocab path

        output_path: string
            Path to save the vocab
    """
    with open(path, 'r') as f, open(output_path, 'w') as out_f:
        for line in f:
            line = line.rstrip()
            if "@@" not in line and len(line) != 1:
                out_f.write(line + '\n')


if __name__ == "__main__":
    path = sys.argv[1]
    output_path = sys.argv[2]
    generate_combiner_data(path, output_path)
