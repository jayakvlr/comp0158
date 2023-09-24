# coding=utf-8
# Copyright 2022 Gen Luo. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from donut.model import  DonutConfig,SwinEncoder
from avalanche.evaluation.metrics.accuracy import Accuracy
from tqdm import tqdm
from timm.models import create_model
from argparse import ArgumentParser
from pythonrepadapter import *
from pythonrepadapter import set_RepAdapter, set_RepWeight
import time

@torch.no_grad()
def test(model, dl):
    model.eval()
    acc = Accuracy()
    pbar = tqdm(dl)
    model = model.cuda()
    for batch in pbar:  # pbar:
        x, y = batch[0].cuda(), batch[1].cuda()
        out = model(x).data
        acc.update(out.argmax(dim=1).view(-1), y)

    return acc.result()




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--model', type=str, default='swin_base_patch4_window7_224_in22k')
    parser.add_argument('--dataset', type=str, default= "JayalekshmiGopakumar/DocLayexp1" )
    parser.add_argument('--method', type=str, default='repblock', choices=['repattn,repblock'])
    parser.add_argument('--scale', type=float, default=1)
    parser.add_argument('--dim', type=int, default=8)
    args = parser.parse_args()
    print(args)
    config = Config(args.method, args.dataset,False)

    if 'swin' in args.model:
        dnconfig = DonutConfig.from_pretrained("naver-clova-ix/donut-base")
        loaded_model = torch.load('artifacts/swin_encoder_from_donut.bin')
        prefix = 'model.'
        for key in list(loaded_model.keys()):
            if key.startswith(prefix):
                new_key = key.replace(prefix, "")
                loaded_model[new_key] = loaded_model.pop(key)
        model = SwinEncoderClass(
            input_size=dnconfig.input_size,
            align_long_axis=dnconfig.align_long_axis,
            encoder_layer=dnconfig.encoder_layer,
            window_size=dnconfig.window_size
        )
        model.load_state_dict(loaded_model)
    else:
        assert NotImplementedError

    # build dataset
    train_dl, test_dl = get_data(args.dataset)

    # running throughput
    model.cuda()
    print('before reparameterizing: ')
    throughput(model)

    # build repadapter
    set_RepAdapter(model, args.method, dim=args.dim, s=args.scale, args=args, set_forward=False)

    # load model
    model.reset_classifier(config.class_num)
    model = load(args.method,'doclaynet', model)

    # fusing repadapter
    set_RepWeight(model, args.method, dim=args.dim, s=args.scale, args=args)

    # running throughput
    model.cuda()
    print(model)
    print('after reparameterizing: ')
    throughput(model)

    # testing loop
    acc = test(model, test_dl)
    print('Accuracy:', acc)

