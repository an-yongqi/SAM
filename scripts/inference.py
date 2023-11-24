from tinysam.modeling import ImageEncoderViT, TinyViT
from functools import partial
import torch
import numpy as np
import tqdm
from torch.backends import cudnn
import os

# Set the device to the fourth GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
repetitions = 300

dummy_input = torch.rand(1, 3, 1024, 1024).to(device)

model = TinyViT(img_size=1024, in_chans=3,
                embed_dims=[64, 128, 160, 320],
                depths=[2, 2, 6, 2],
                num_heads=[2, 4, 5, 10],
                window_sizes=[7, 7, 14, 7],
                mlp_ratio=4.,
                drop_rate=0.,
                drop_path_rate=0.0,
                use_checkpoint=False,
                mbconv_expand_ratio=4.0,
                local_conv_size=3,
                layer_lr_decay=0.8
            ).to(device)

# model = ImageEncoderViT(
#             depth=32,
#             embed_dim=1280,
#             img_size=1024,
#             mlp_ratio=4,
#             norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
#             num_heads=16,
#             patch_size=16,
#             qkv_bias=True,
#             use_rel_pos=True,
#             global_attn_indexes=[7, 15, 23, 31],
#             window_size=14,
#             out_chans=256,
#         ).to(device)

# 预热, GPU 平时可能为了节能而处于休眠状态, 因此需要预热
print('warm up ...\n')
with torch.no_grad():
    for _ in range(100):
        _ = model(dummy_input)


# 设置用于测量时间的 cuda Event, 这是PyTorch 官方推荐的接口,理论上应该最靠谱
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
# 初始化一个时间容器
timings = np.zeros((repetitions, 1))

print('testing ...\n')
with torch.no_grad():
    for rep in tqdm.tqdm(range(repetitions)):
        starter.record()
        _ = model(dummy_input)
        ender.record()
        torch.cuda.synchronize() # 等待GPU任务完成
        curr_time = starter.elapsed_time(ender) # 从 starter 到 ender 之间用时,单位为毫秒
        timings[rep] = curr_time

avg = timings.sum()/repetitions
print('\navg={}\n'.format(avg))