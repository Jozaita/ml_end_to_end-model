from ml_end_to_end.models.transformations import Transformation
from ml_end_to_end.utils.io_utils import translate_gcs_dir_to_local
from torch import nn
from transformers import AutoConfig,AutoModel,BatchEncoding
from transformers.modeling_outputs import BaseModelOutputWithPooling

class BackBone(nn.Module):
    def __init__(self, transformation:Transformation) -> None:
        super().__init__()
        self.transformation = transformation

    def get_transformation(self)->Transformation:
        return self.transformation


class HuggingFaceBackbone(BackBone):
    def __init__(self,pretrained_model_name_or_path:str,transformation:Transformation,pretrained:bool = False)->None:
        super().__init__(transformation)

        self.pretrained = pretrained
        self.transformation = transformation 
        self.backbone = self.get_backbone(pretrained_model_name_or_path)
        

    def forward(self,encodings:BatchEncoding)->BaseModelOutputWithPooling:
        output:BaseModelOutputWithPooling = self.backbone(**encodings)
        return output


    def get_backbone(self,pretrained_model_name_or_path:str):
        path = translate_gcs_dir_to_local(pretrained_model_name_or_path)
        config = AutoConfig.from_pretrained(path)
        if self.pretrained:
            return AutoModel(path=path,config=config)
        
        return AutoModel.from_config(config)


