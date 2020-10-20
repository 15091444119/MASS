"""
Extract log from old file and write into tensorboard
"""

import argparse
import json
from tensorboardX import SummaryWriter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("saved_dir")
    args = parser.parse_args()

    writer = SummaryWriter(args.saved_dir)

    with open(args.log, 'r') as f:
        for line in f:
            if "__log__" in line and "epoch" in line:
                line = line.rstrip()
                log_str = line[line.index('{'):]
                log = json.loads(log_str)
                epoch = log["epoch"]
                if epoch == -1:
                    continue
                if epoch == 2:
                    print(log["valid-zh-combiner"])
                for key in log:
                    if key != "epoch":
                        writer.add_scalar(key, log[key], global_step=epoch)
