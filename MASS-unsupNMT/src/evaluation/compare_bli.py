# compare two bli translation
import argparse


def read_dict(path):
    """
    Read the ground truth dic or a bli results produced by BLI.eval, if the later
    params:
        path(str): path of the given dic
    """
    dic = {}
    with open(path, 'r') as f:
        for line in f:
            src, tgt = line.rstrip().split()
            if src not in dic:
                dic[src] = [tgt]
            else:
                dic[src].append(tgt)
    return dic


def have_translation(hyp_list, ref_list):
    """ If a word in hyp_list is in ref_list, return True, else False"""
    return len(set(hyp_list).intersection((set(ref_list)))) != 0


def compare_bli(bli_dict1, bli_dict2, ground_truth_dict, topk=5):
    """
    Compare two bli dictionary
    Params:
        bli_dict1(dict):
        bli_dict2(dict):
        ground_truth_dict(dict):
        topk: topk items used in bli_dict1 and bli_dict2 for compare (assume the list in dict is sorted)
    Returns:
        right_right, right_wrong, wrong_right, wrong_wrong

    """
    # have the same source words
    for src in bli_dict1:
        assert src in bli_dict2
        assert src in ground_truth_dict

    right_right, right_wrong, wrong_right, wrong_wrong = 0, 0, 0, 0

    for src, tgt_list1 in bli_dict1.items():
        ref_list = ground_truth_dict[src]
        tgt_list2 = bli_dict2[src]
        ans1 = have_translation(tgt_list1[:topk], ref_list)
        ans2 = have_translation(tgt_list2[:topk], ref_list)
        if ans1 and ans2:
            right_right += 1
        elif ans1 and not ans2:
            right_wrong += 1
        elif not ans1 and ans2:
            wrong_right += 1
        else:
            wrong_wrong += 1

    return right_right, right_wrong, wrong_right, wrong_wrong


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--bli1")
    parser.add_argument("--bli2")
    parser.add_argument("--ref")
    parser.add_argument("--topk", type=int, help="topk hyp to use for evaluation")
    args = parser.parse_args()

    bli_dict1 = read_dict(args.bli1)
    bli_dict2 = read_dict(args.bli2)
    ground_truth_dict = read_dict(args.ref)

    right_right, right_wrong, wrong_right, wrong_wrong = compare_bli(bli_dict1, bli_dict2, ground_truth_dict, args.topk)

    print("Right_right: {}\nRight_wrong: {}\nWrong_right: {}\nWrong_wrong: {}".format(right_right, right_wrong, wrong_right, wrong_wrong))
