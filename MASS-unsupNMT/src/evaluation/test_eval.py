import torch
from .eval_representation import get_sen_representation

def test_get_sen_representation():
    encoded = torch.tensor([[[1.0, 2.0], [2.0, 3.0]], [[1.0, 2.0], [2.0, 3.0]]]).cuda()
    length = torch.tensor([1, 2]).cuda()
    ans_avg = torch.tensor([[1.0, 2.0],[1.5, 2.5]]).cuda()
    ans_max = torch.tensor([[1.0, 2.0],[2.0, 3.0]]).cuda()
    ans_cls = torch.tensor([[1.0, 2.0],[1.0, 2.0]]).cuda()
    assert (get_sen_representation(encoded, length, "avg") == ans_avg).all()
    assert (get_sen_representation(encoded, length, "max") == ans_max).all()
    assert (get_sen_representation(encoded, length, "cls") == ans_cls).all()

      
if __name__ == "__main__":
    test_get_sen_representation()
