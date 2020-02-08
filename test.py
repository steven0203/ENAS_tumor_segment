import torch
from models import isensee2017_model


test=torch.randn(1,4,128,128,128)
model=isensee2017_model()
tmp=model(test)