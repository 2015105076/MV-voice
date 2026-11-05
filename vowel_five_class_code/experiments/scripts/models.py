"""Protocol branches and valid-length TF forwards, including official BEATs bridge."""
import sys

import torch
from torch import nn
from torchvision.models import resnet18

from freeze_data import ROOT


class MaskedGroupNorm(nn.Module):
    def __init__(self,channels,groups=8,eps=1e-5):
        super().__init__()
        self.groups,self.eps = groups,eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self,x,mask):
        b,c,t = x.shape
        grouped = x.float().reshape(b,self.groups,c//self.groups,t)
        valid = mask[:,None,None,:].float()
        count = valid.sum(-1,keepdim=True)*(c//self.groups)
        mean = (grouped*valid).sum((2,3),keepdim=True)/count.clamp_min(1)
        variance = ((grouped-mean).square()*valid).sum((2,3),keepdim=True)/count.clamp_min(1)
        out = ((grouped-mean)/torch.sqrt(variance+self.eps)).reshape(b,c,t).to(x.dtype)
        return (out*self.weight[None,:,None]+self.bias[None,:,None])*mask[:,None,:]


class TemporalBlock(nn.Module):
    def __init__(self,inputs,outputs):
        super().__init__()
        self.conv1 = nn.Conv1d(inputs,outputs,5,padding=2)
        self.norm1 = MaskedGroupNorm(outputs)
        self.conv2 = nn.Conv1d(outputs,outputs,5,padding=2)
        self.norm2 = MaskedGroupNorm(outputs)
        self.skip = nn.Conv1d(inputs,outputs,1) if inputs!=outputs else nn.Identity()
        self.activation = nn.GELU()

    def forward(self,x,mask):
        valid = mask[:,None,:]
        x = x*valid
        out = self.activation(self.norm1(self.conv1(x),mask))*valid
        out = self.norm2(self.conv2(out),mask)
        return self.activation(out+self.skip(x))*valid


class TemporalEncoder(nn.Module):
    def __init__(self,dropout):
        super().__init__()
        self.blocks = nn.ModuleList([TemporalBlock(1,32),TemporalBlock(32,64)])
        self.project = nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.Dropout(dropout))

    def forward(self,x,mask):
        x = x[:,None,:]
        for block in self.blocks:
            x = block(x,mask)
        x = x.float()
        count = mask.sum(-1,keepdim=True).clamp_min(1)
        mean = (x*mask[:,None,:]).sum(-1)/count
        var = ((x-mean[:,:,None]).square()*mask[:,None,:]).sum(-1)/count
        return self.project(torch.cat([mean,torch.sqrt(var+1e-8)],dim=-1))


def load_beats():
    vendor = str(ROOT/'vendor'/'beats')
    if vendor not in sys.path:
        sys.path.insert(0,vendor)
    from BEATs import BEATs,BEATsConfig
    checkpoint = torch.load(ROOT/'pretrained'/'BEATs_iter3.pt',map_location='cpu',weights_only=True)
    cfg = BEATsConfig(checkpoint['cfg'])
    assert not cfg.finetuned_model
    model = BEATs(cfg)
    model.load_state_dict(checkpoint['model'],strict=True)
    return model


def beats_from_logmel(model,L):
    """Exact official extract_features after its preprocess; no padded frames supplied."""
    normalized = (L-15.41663)/(2*6.55582)
    padding = torch.zeros(normalized.shape[:2],device=L.device,dtype=torch.bool)
    features = model.patch_embedding(normalized.unsqueeze(1))
    features = features.reshape(features.shape[0],features.shape[1],-1).transpose(1,2)
    features = model.layer_norm(features)
    padding = model.forward_padding_mask(features,padding)
    if model.post_extract_proj is not None:
        features = model.post_extract_proj(features)
    x = model.dropout_input(features)
    x,_ = model.encoder(x,padding_mask=padding)
    return x,padding


class MultiViewModel(nn.Module):
    def __init__(self,backbone,view_string,dropout=.3,fusion='concat'):
        super().__init__()
        self.backbone_id,self.views,self.fusion = backbone,view_string.split('+'),fusion
        if 'TF' in self.views:
            if backbone=='resnet18_imagenet':
                self.backbone = resnet18(weights=None)
                self.backbone.load_state_dict(torch.load(ROOT/'pretrained'/'resnet18-f37072fd.pth',map_location='cpu',weights_only=True))
                self.backbone.fc = nn.Identity()
                output = 512
            else:
                self.backbone = load_beats()
                output = self.backbone.cfg.encoder_embed_dim
                for p in self.backbone.parameters():
                    p.requires_grad_(False)
                if backbone=='beats_last2':
                    for module in [*self.backbone.encoder.layers[-2:],self.backbone.encoder.layer_norm]:
                        for p in module.parameters():
                            p.requires_grad_(True)
                elif backbone!='beats_frozen':
                    raise ValueError(backbone)
            self.tf_projection = nn.Sequential(nn.Linear(output,128),nn.GELU(),nn.Dropout(dropout))
        if 'FE' in self.views:
            self.fe = nn.Sequential(nn.Linear(128,128),nn.GELU(),nn.LayerNorm(128),nn.Dropout(dropout),
                                    nn.Linear(128,128),nn.GELU(),nn.LayerNorm(128))
        if 'TE' in self.views:
            self.te = TemporalEncoder(dropout)
        if len(self.views)==1:
            self.head = nn.Sequential(nn.Dropout(dropout),nn.Linear(128,5))
        elif fusion=='concat':
            self.head = nn.Sequential(nn.Linear(128*len(self.views),128),nn.GELU(),nn.Dropout(dropout),nn.Linear(128,5))
        elif fusion=='gated':
            self.score = nn.Sequential(nn.Linear(128,32),nn.GELU(),nn.Linear(32,1))
            self.head = nn.Linear(128,5)
        else:
            raise ValueError(fusion)

    def train(self,mode=True):
        super().train(mode)
        if 'TF' in self.views and self.backbone_id.startswith('beats'):
            self.backbone.eval()
            if self.backbone_id=='beats_last2':
                for module in [*self.backbone.encoder.layers[-2:],self.backbone.encoder.layer_norm]:
                    module.train(mode)
        return self

    def tf_embedding(self,L):
        if self.backbone_id=='resnet18_imagenet':
            return self.backbone(L[None,None].expand(-1,3,-1,-1))
        tokens,mask = beats_from_logmel(self.backbone,L[None])
        return (tokens*(~mask)[:,:,None]).sum(1)/(~mask).sum(1,keepdim=True)

    def forward(self,items):
        results = []
        # No padding reaches convolutions or BatchNorm; each recording keeps its T.
        for item in items:
            embeddings = []
            for view in self.views:
                if view=='TF':
                    tf = item.get('frozen_tf_embedding')
                    if tf is None:
                        tf = self.tf_embedding(item['TF'])
                    else:
                        tf = tf[None]
                    embeddings.append(self.tf_projection(tf))
                elif view=='FE':
                    embeddings.append(self.fe(item['FE'][None]))
                elif view=='TE':
                    te = item['TE'][None]
                    embeddings.append(self.te(te,torch.ones_like(te,dtype=torch.bool)))
            if len(embeddings)==1:
                combined = embeddings[0]
            elif self.fusion=='concat':
                combined = torch.cat(embeddings,dim=-1)
            else:
                stack = torch.stack(embeddings,dim=1)
                weights = self.score(stack).softmax(1)
                combined = (stack*weights).sum(1)
            results.append(self.head(combined))
        return torch.cat(results,dim=0)
