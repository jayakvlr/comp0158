#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pytorch_lightning as pl
from torch.optim import AdamW
from timm.scheduler.cosine_lr import CosineLRScheduler
from argparse import ArgumentParser
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from donut.model import DonutConfig
from timm.models.swin_transformer import SwinTransformer
from timm.models import create_model
from avalanche.evaluation.metrics.accuracy import Accuracy
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch
import torch.nn as nn
import numpy as np
import time

from argparse import ArgumentParser
from pytorch_lightning.plugins import CheckpointIO
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LambdaLR
import math
import os
import re
from typing import Any, List, Optional, Union
from donut.model import  DonutConfig,SwinEncoder
import numpy as np
import PIL
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageOps
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.swin_transformer import SwinTransformer
from torchvision import transforms
from torchvision.transforms.functional import resize, rotate
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

import os
import numpy as np
from PIL import Image
import torch
from torch.utils import data

class ImageFilelist(data.Dataset):
    def __init__(self, data, transform=None, target_transform=None):
        self.data = data
        self.transform = transform
        self.target_transform = target_transform
  

    def __getitem__(self, index):
        image_data, target = self.data[index]['image'],self.data[index]['label']

        # Convert image_data to a PIL Image or provide proper handling based on the data format
        img = Image.fromarray(np.uint8(image_data))

        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        print(target)
        return img, target

    def __len__(self):
        return len(self.data)


# In[3]:


import torch
from datasets import load_dataset
from torchvision import transforms

def get_data(name, evaluate=True, batch_size=8, shot=1, seed=1, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    if evaluate:
        split = 'test'
    else:
        split = 'train'

    # Load the dataset from Hugging Face
    dataset = load_dataset(name)


    train_transform = transforms.Compose([
                transforms.Resize((2560, 1920), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
    val_transform = transforms.Compose([
                transforms.Resize((2560, 1920), interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])

    if evaluate:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(data=dataset['train'],
                transform=train_transform),
            batch_size=batch_size, shuffle=True, drop_last=True,
            num_workers=4, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            ImageFilelist(data=dataset['validation'],
                transform=val_transform),
            batch_size=8, shuffle=False,
            num_workers=4, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(data=dataset['validation'],
                transform=train_transform),
            batch_size=batch_size, shuffle=True, drop_last=True,
            num_workers=4, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            ImageFilelist(data=dataset['test'],
                transform=val_transform),
            batch_size=256, shuffle=False,
            num_workers=4, pin_memory=True)
    return train_loader, val_loader



def load(method, dataset, model):
    model = model.cpu()
    st = torch.load('./models/%s/%s.pt'%(method, dataset))
    model.load_state_dict(st, False)
    return model



class Config:
    def __init__(self,method,dataset,few_shot):
        self.name = "cifar"
        self.class_num = 6
        self.scale = 0.1


class SwinEncoderClass(SwinTransformer):

    def __init__(
        self,
        input_size: List[int],
        align_long_axis: bool,
        window_size: int,
        encoder_layer: List[int],
        name_or_path: Union[str, bytes, os.PathLike] = None,
    ):
        # Call the __init__ method of the parent class to initialize all methods and attributes
        super().__init__(
            img_size=input_size,
            depths=encoder_layer,
            window_size=window_size,
            patch_size=4,
            embed_dim=128,
            num_heads=[4, 8, 16, 32],
            num_classes=0,
        )

        # Modify any additional attributes or methods specific to your custom architecture here
        self.input_size = input_size
        self.align_long_axis = align_long_axis
        self.window_size = window_size
        self.encoder_layer = encoder_layer
        # Freeze the weights of the swin_encoder
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        return super().forward(x)

        
    def prepare_input(self, img: PIL.Image.Image, random_padding: bool = False) -> torch.Tensor:
        """
        Convert PIL Image to tensor according to specified input_size after following steps below:
            - resize
            - rotate (if align_long_axis is True and image is not aligned longer axis with canvas)
            - pad
        """
        img = img.convert("RGB")
        if self.align_long_axis and (
            (self.input_size[0] > self.input_size[1] and img.width > img.height)
            or (self.input_size[0] < self.input_size[1] and img.width < img.height)
        ):
            img = rotate(img, angle=-90, expand=True)
        img = resize(img, min(self.input_size))
        img.thumbnail((self.input_size[1], self.input_size[0]))
        delta_width = self.input_size[1] - img.width
        delta_height = self.input_size[0] - img.height
        if random_padding:
            pad_width = np.random.randint(low=0, high=delta_width + 1)
            pad_height = np.random.randint(low=0, high=delta_height + 1)
        else:
            pad_width = delta_width // 2
            pad_height = delta_height // 2
        padding = (
            pad_width,
            pad_height,
            delta_width - pad_width,
            delta_height - pad_height,
        )
        return self.to_tensor(ImageOps.expand(img, padding))




# In[18]:


class RepTraining(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = Config(args.method, args.dataset, args.few_shot)
        self.dnconfig = DonutConfig.from_pretrained("naver-clova-ix/donut-base")
        self.model = self.configure_model()
        if self.args.few_shot:
            self.mixup_fn=Mixup(
                    mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
                    prob=1.0, switch_prob=0.5, mode='batch',
                    label_smoothing=0.1, num_classes=config.class_num)
            self.criterion = SoftTargetCrossEntropy()
        else:
            self.mixup_fn=None
            self.criterion = torch.nn.CrossEntropyLoss()
            self.best_acc = 0

    def configure_model(self):
        loaded_model = torch.load('artifacts/swin_encoder_from_donut.bin')
        prefix = 'model.'
        for key in list(loaded_model.keys()):
            if key.startswith(prefix):
                new_key = key.replace(prefix, "")
                loaded_model[new_key] = loaded_model.pop(key)
        self.model = SwinEncoderClass(
            input_size=self.dnconfig.input_size,
            align_long_axis=self.dnconfig.align_long_axis,
            encoder_layer=self.dnconfig.encoder_layer,
            window_size=self.dnconfig.window_size
        )

        self.model.load_state_dict(loaded_model)
        for param in self.model.parameters():
            param.requires_grad = False
        #throughput(self.model)
        set_RepAdapter(self.model, self.args.method, dim=self.args.dim, s=self.config.scale if self.args.scale==0 else self.args.scale, args=args)

        self.model.reset_classifier(self.config.class_num)
                # Make layers with 'adapter' or 'conv' in their names trainable
        #throughput(self.model)


        return self.model

    def forward(self, x):
        return self.model(x)



    def change_tensor(self, target_shape, image_tensor):
        current_shape = image_tensor.shape
        pad_height = target_shape[0] - current_shape[2]
        pad_width = target_shape[1] - current_shape[3]
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        return torch.nn.functional.pad(image_tensor, (pad_left, pad_right, pad_top, pad_bottom), 'constant')

    def train_dataloader(self):
        train_dl, _ = get_data(self.args.dataset, batch_size=4,evaluate=True)
        return train_dl

    def val_dataloader(self):
        _, val_dl = get_data(self.args.dataset, batch_size=4,evaluate=False)
        return val_dl

    def test_dataloader(self):
        _, test_dl = get_data(self.args.dataset, batch_size=1,evaluate=False)
        return test_dl
    @staticmethod
    def cosine_scheduler(optimizer, training_steps, warmup_steps):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            progress = current_step - warmup_steps
            progress /= max(1, training_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(optimizer, lr_lambda)

    def configure_optimizers(self):
        max_epochs=10
        trainable_params = [p for n, p in self.model.named_parameters() if 'conv' in n or 'rep' in n]
        optimizer = AdamW(trainable_params, lr=self.args.lr, weight_decay=self.args.wd)
        max_iter = None

        if int(max_epochs) > 0:
            max_iter = max_epochs

        assert max_iter is not None
        scheduler = {
            "scheduler": self.cosine_scheduler(optimizer, max_iter, 5),
            "name": "learning_rate",
            "interval": "step",
        }
        return [optimizer], [scheduler]


    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.change_tensor(self.dnconfig.input_size, x)
        if self.mixup_fn is not None:
                x,y=mixup_fn(x,y)
        out = self(x)
        loss = self.criterion(out, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = self.change_tensor(self.dnconfig.input_size, x)
        out = self(x)
        pred = out.argmax(dim=1)
        acc = Accuracy()
        acc.update(pred.view(-1), y)
        self.log('val_acc', acc.result(), on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = self.change_tensor(self.dnconfig.input_size, x)
        out = self(x)
        pred = out.argmax(dim=1)
        acc = Accuracy()
        acc.update(pred.view(-1), y)
        self.log('test_acc', acc.result())






# In[13]:


import torch
from torch import nn
import timm

def forward_swin_block_adapter(self, x):
    H, W = self.input_resolution
    B, L, C = x.shape
    assert L == H * W, "input feature has wrong size"

    shortcut = x
    x = self.adapter_attn(self.norm1(x))
    x = x.view(B, H, W, C)

    # cyclic shift
    if self.shift_size > 0:
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
    else:
        shifted_x = x

    # partition windows
    x_windows = timm.models.swin_transformer.window_partition(shifted_x,
                                                              self.window_size)  # nW*B, window_size, window_size, C
    x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

    # W-MSA/SW-MSA
    attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

    # merge windows
    attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
    shifted_x = timm.models.swin_transformer.window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

    # reverse cyclic shift
    if self.shift_size > 0:
        x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
    else:
        x = shifted_x
    x = x.view(B, H * W, C)

    # FFN
    x = shortcut + self.drop_path(x)
    x = x + self.drop_path(self.mlp(self.adapter_mlp(self.norm2(x))))
    return x


def forward_swin_attn_adapter(self, x):
    H, W = self.input_resolution
    B, L, C = x.shape
    assert L == H * W, "input feature has wrong size"

    shortcut = x
    x = self.adapter_attn(self.norm1(x))
    x = x.view(B, H, W, C)

    # cyclic shift
    if self.shift_size > 0:
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
    else:
        shifted_x = x

    # partition windows
    x_windows = timm.models.swin_transformer.window_partition(shifted_x,
                                                              self.window_size)  # nW*B, window_size, window_size, C
    x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

    # W-MSA/SW-MSA
    attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

    # merge windows
    attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
    shifted_x = timm.models.swin_transformer.window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

    # reverse cyclic shift
    if self.shift_size > 0:
        x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
    else:
        x = shifted_x
    x = x.view(B, H * W, C)

    # FFN
    x = shortcut + self.drop_path(x)
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    return x

class RepAdapter(nn.Module):
    """ Pytorch Implemention of RepAdapter for 1d tensor"""

    def __init__(
            self,
            in_features=768,
            hidden_dim=8,
            groups=2,
            scale=1
    ):
        super().__init__()
        self.conv_A=nn.Conv1d(in_features,hidden_dim,1,groups=1,bias=True)
        self.conv_B = nn.Conv1d(hidden_dim, in_features, 1, groups=groups, bias=True)
        self.dropout=nn.Dropout(0.1)
        self.groups=groups
        self.scale=scale

        nn.init.xavier_uniform_( self.conv_A.weight)
        nn.init.zeros_(self.conv_A.bias)
        nn.init.zeros_(self.conv_B.weight)
        nn.init.zeros_(self.conv_B.bias)
    def forward(self, x):
        x=x.transpose(1,2)
        x=self.conv_B(self.dropout(self.conv_A(x)))*self.scale+x
        x=x.transpose(1,2).contiguous()
        return x


class RepAdapter2D(nn.Module):
    """ Pytorch Implemention of RepAdapter for 2d tensor"""

    def __init__(
            self,
            in_features=768,
            hidden_dim=8,
            groups=2,
            scale=1
    ):
        super().__init__()
        self.conv_A = nn.Conv2d(in_features, hidden_dim, 1, groups=1, bias=True)
        self.conv_B = nn.Conv2d(hidden_dim, in_features, 1, groups=groups, bias=True)
        self.dropout = nn.Dropout(0.1)
        self.groups = groups
        self.scale = scale

        nn.init.xavier_uniform_(self.conv_A.weight)
        nn.init.zeros_(self.conv_A.bias)
        nn.init.zeros_(self.conv_B.weight)
        nn.init.zeros_(self.conv_B.bias)

    def forward(self, x):
        x = self.conv_B(self.dropout(self.conv_A(x))) * self.scale + x
        return x

def reparameterize(Wa,Wb,Ba,Bb,scale=1,do_residual=False):
    bias = 0
    id_tensor=0
    if Ba is not None:
        bias=Ba@Wb
    if Bb is not None:
        bias=bias+Bb
    if do_residual:
        id_tensor=torch.eye(Wa.shape[0],Wb.shape[1]).to(Wa.device)
    weight = Wa @ Wb*scale + id_tensor
    return weight.T,bias*scale if isinstance(bias,torch.Tensor) else None

def sparse2dense(weight,groups):
    d,cg=weight.shape
    dg=d//groups
    weight=weight.view(groups,dg,cg).transpose(1,2)
    new_weight=torch.zeros(cg*groups,d,device=weight.device,dtype=weight.dtype)
    for i in range(groups):
        new_weight[i*cg:(i+1)*cg,i*dg:(i+1)*dg]=weight[i]
    return new_weight.T


def set_RepAdapter(model, method, dim=8, s=1, args=None,set_forward=True):

    if method == 'repblock':
        for _ in model.children():
            if type(_) == timm.models.swin_transformer.SwinTransformerBlock:
                _.adapter_attn = RepAdapter(in_features=_.dim,hidden_dim=dim,scale=s)
                _.adapter_mlp = RepAdapter(in_features=_.dim,hidden_dim=dim,scale=s)
                _.s = s
                bound_method = forward_swin_block_adapter.__get__(_, _.__class__)
                setattr(_, 'forward', bound_method)
 
            elif len(list(_.children())) != 0:
                set_RepAdapter(_, method, dim, s,args=args,set_forward=set_forward)

    else:
        for _ in model.children():
            if type(_) == timm.models.swin_transformer.SwinTransformerBlock:
                _.adapter_attn =  RepAdapter(in_features=_.dim,hidden_dim=dim,scale=s)
                _.s = s
                bound_method = forward_swin_attn_adapter.__get__(_, _.__class__)
                setattr(_, 'forward', bound_method)
            elif len(list(_.children())) != 0:
                set_RepAdapter(_, method, dim, s, args=args, set_forward=set_forward)

def set_RepWeight(model, method, dim=8, s=1, args=None):
    if method == 'repblock':
        for _ in model.children():
            if type(_) == timm.models.vision_transformer.Block or type(_) == timm.models.swin_transformer.SwinTransformerBlock:
                if _.adapter_attn.groups>1:
                    weight_B=sparse2dense(_.adapter_attn.conv_B.weight.squeeze(),_.adapter_attn.groups)
                else:
                    weight_B=_.adapter_attn.conv_B.weight.squeeze()
                attn_weight,attn_bias=reparameterize(_.adapter_attn.conv_A.weight.squeeze().T,weight_B.T,
                                        _.adapter_attn.conv_A.bias,_.adapter_attn.conv_B.bias,_.s,do_residual=True)
                qkv_weight,qkv_bias=reparameterize(attn_weight.T,_.attn.qkv.weight.T,
                                                attn_bias, _.attn.qkv.bias)
                with torch.no_grad():
                    _.attn.qkv.weight.copy_(qkv_weight)
                    _.attn.qkv.bias.copy_(qkv_bias)

@torch.no_grad()
def throughput(model,img_size=[2560, 1920],bs=1):
    with torch.no_grad():
        x = torch.randn(bs, 3, img_size[0], img_size[1]).cuda()
        batch_size=x.shape[0]
        # model=create_model('vit_base_patch16_224_in21k', checkpoint_path='./ViT-B_16.npz', drop_path_rate=0.1)
        model.eval()
        for i in range(50):
            model(x)
        torch.cuda.synchronize()
        print(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(x)
        torch.cuda.synchronize()
        tic2 = time.time()
        print(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        MB = 1024.0 * 1024.0
        print('memory:', torch.cuda.max_memory_allocated() / MB)


class CustomCheckpointIO(CheckpointIO):
    def save_checkpoint(self, checkpoint, path, storage_options=None):
        # Ensure the parent directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Your save_checkpoint code here
        del checkpoint["state_dict"]
        print(path)
        torch.save(checkpoint, path)

    def load_checkpoint(self, path, storage_options=None):
        # Ensure the parent directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = torch.load(path + "artifacts.ckpt")
        state_dict = torch.load(path + "pytorch_model.bin")
        checkpoint["state_dict"] = {"model." + key: value for key, value in state_dict.items()}
        return checkpoint

    def remove_checkpoint(self, path) -> None:
        # Ensure the parent directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        return super().remove_checkpoint(path)

@torch.no_grad()
def save(method, dataset, model):
    model.eval()
    model = model.cpu()
    trainable = {}
    for n, p in model.named_parameters():
        if 'adapter' in n or 'head' in n:
            trainable[n] = p.data
        # Define the directory where you want to save the model and log file
    save_dir = './models/{}/'.format(method)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(trainable, os.path.join(save_dir, '{}.pt'.format(dataset)))
    
def load(method, dataset, model):
    model = model.cpu()
    st = torch.load('./models/%s/%s.pt'%(method, dataset))
    model.load_state_dict(st, False)
    return model

class MainConfig:
    def __init__(self):
        self.seed = 42
        self.lr = 1e-3
        self.wd = 1e-4
        self.model = 'swin_base_patch4_window7_224_in22'
        self.dataset = 'JayalekshmiGopakumar/DocLayexp1'
        self.method = 'repblock'
        self.scale = 0
        self.dim = 8
        self.few_shot = False
        self.shots = 1


if __name__ == '__main__':
    args=MainConfig()
    print('device counts',torch.cuda.device_count())
    model = RepTraining(args)
    print(model)
    custom_ckpt=CustomCheckpointIO()
    gradient_accumulation_steps = 20
    trainer = pl.Trainer(
            num_nodes=1,
            devices=torch.cuda.device_count(),
            strategy='ddp',
            accelerator="gpu",
            plugins=custom_ckpt,
            max_epochs=20,
            max_steps=-1,
            precision=16,
            val_check_interval=1.0,
            check_val_every_n_epoch=1,
            gradient_clip_val=1.0,
            num_sanity_val_steps=0,
            accumulate_grad_batches=gradient_accumulation_steps
        )

    trainer.fit(model)
    save(method=args.method,model=model,dataset='doclaynet')









