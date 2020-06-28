from .eval_attention import draw_attention
import numpy as np

def test_draw_attention():
    draw_attention(np.array([[0.1, 0.5, 0.4],[0.5,0.5, 0]]), ["src_1", "src_2"], ["tgt_1", "tgt_2", "tgt_3"], "./tmp.txt")

if __name__ == "__main__":
    test_draw_attention()