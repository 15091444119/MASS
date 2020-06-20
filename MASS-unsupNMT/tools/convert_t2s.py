import sys
import opencc

if __name__ == "__main__":
    converter = opencc.OpenCC('t2s.json')
    for line in sys.stdin:
        line = converter.convert(line.rstrip())
        print(line)
