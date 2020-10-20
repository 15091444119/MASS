"""
Analysis translation results
"""

import sys


if __name__ == "__main__":
    path1 = sys.argv[1]
    path2 = sys.argv[2]

    with open(path1) as f1:
        with open(path2) as f2:
            path1_right_src = {}
            while(True):
                line = f1.readline()
                if line == "":
                    break
                src, tgt, state = line.rstrip().split()
                if state == "True":
                    if src not in path1_right_src:
                        path1_right_src[src] = [tgt]
                    else:
                        path1_right_src[src].append(tgt)
                for i in range(9):
                    f1.readline()

            while(True):
                line = f2.readline()
                if line == "":
                    break
                src, tgt, state = line.rstrip().split()
                if state == "True":
#                    if src not in path1_right_src:
                        print(line)
                for i in range(9):
                    f2.readline()




