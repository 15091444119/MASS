"""
cheat means we use original whole word representation as combined results
"""


from .mass import set_model_mode
from .explicit_split import ExplicitSplitCombineTool, ExplicitSplitEncoderBatch
from ..combine_utils import CheatCombineTool
from .common_combine import BaseCombinerEncoder, DecodeInputBatch, BaseEncoder, BaseSeq2Seq, EncoderInputs, CommonEncoder, LossDecodingBatch, GenerateDecodeBatch



