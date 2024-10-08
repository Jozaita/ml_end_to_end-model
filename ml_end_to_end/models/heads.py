from torch import nn,Tensor

class Head(nn.Module):
    pass

class SoftmaxHead(Head):
    def __init__(self,in_features:int,out_features:int,dim:int=1)->None:
        super().__init__()

        self.head = nn.Sequential(nn.Linear(in_features, out_features),nn.Softmax(dim=dim))
    
    def forward(self, x:Tensor)->Tensor:
        return self.head(x)

class SigmoidHead(Head):
    def __init__(self,in_features:int,out_features:int)->None:
        super().__init__()

        self.head = nn.Sequential(nn.Linear(in_features, out_features),nn.Sigmoid())
    
    def forward(self, x:Tensor)->Tensor:
        return self.head(x)
