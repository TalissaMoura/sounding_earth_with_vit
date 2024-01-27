import torch
import torch.nn as nn
import numpy as np
import timm

class VisionTransformersEncoder(nn.Module):
  def __init__(self,batch_size=32,output_dim=128,model_id='google/vit-base-patch16-224-in21k',**kwargs):
    super(VisionTransformersEncoder, self).__init__(**kwargs)
    self.vit_model = timm.create_model(model_id, pretrained=True, num_classes=0,global_pool='')
    self.batch_size=batch_size
    self.output_dim=output_dim
    self.dropout = nn.Dropout(p=0.2)
    
    self.first_avg_pool_layer = nn.AvgPool1d(kernel_size=3,stride=1) 

    # Define conv1d layer
    size_input_channels = self.vit_model.pos_embed.size()[1]
    assert size_input_channels > self.batch_size
    self.first_conv1d_layer = nn.Conv1d(in_channels=size_input_channels,out_channels=self.batch_size,kernel_size=3,stride=2)
    assert (self.first_conv1d_layer.out_channels == self.batch_size) or (self.first_conv1d_layer.out_channels == 1)
    
    # Define linear layer
    
    if self.vit_model.num_features == 768:
      self.linear = nn.Linear(381,self.output_dim)
    elif self.vit_model.num_features == 1024:
      self.linear = nn.Linear(509,self.output_dim)
 
   
  def forward(self,value):
    embeddings = self.vit_model(value)
    first_conv_layer = self.first_conv1d_layer(embeddings)
    first_avg_pool_layer = self.first_avg_pool_layer(first_conv_layer)

    outputs = self.dropout(first_avg_pool_layer)
    final_output = self.linear(outputs)
    return final_output