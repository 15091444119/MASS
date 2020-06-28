from .eval_attention import draw_attention
import numpy as np

def test_draw_attention():
    draw_attention(np.array([[0.1, 0.5, 0.4],[0.5,0.5, 0]]), target_tokens=["tgt_1", "tgt_2"], source_tokens=["src_1", "src_2", "src_3"], output_path="./tmp.jpg")

if __name__ == "__main__":
    test_draw_attention()