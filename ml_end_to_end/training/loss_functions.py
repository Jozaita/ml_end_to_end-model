from typing import Optional
from torch import Tensor, nn
import torch.nn.functional as F

class LossFunction(nn.Module):
    pass

class BCEWithLogitsLoss(LossFunction):
    def __init__(self,
                 reduction:str="mean")->None:
        super().__init__()
        self.reduction = reduction

    def forward(self, x:Tensor, targets:Tensor,pos_weight:Optional[Tensor] = None)->Tensor:

        return F.binary_cross_entropy_with_logits(x, targets, reduction=self.reduction,pos_weight=pos_weight)