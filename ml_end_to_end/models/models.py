from abc import abstractmethod
from typing import Optional
from ml_end_to_end.models.transformations import Transformation
from ml_end_to_end.models.adapters import Adapter
from ml_end_to_end.models.backbones import BackBone
from ml_end_to_end.models.heads import Head
from torch import nn,Tensor
from transformers import BatchEncoding


class Model(nn.Module):
    @abstractmethod
    def get_transformation(self)->Transformation:
        ...


class BinaryTextClassificationModel(Model):
    def __init__(self, backbone:BackBone, head:Head,adapter:Optional[Adapter])->None:
        super().__init__()

        self.backbone = backbone
        self.head = head
        self.adapter = adapter 

    def forward(self,encodings:BatchEncoding)->Tensor:
        output = self.backbone(encodings)
        if self.adapter is not None:
            output = self.adapter(output)
        output = self.head(output)
        return output
    
    def get_transformation(self) -> Transformation:
        return self.backbone.get_transformation()