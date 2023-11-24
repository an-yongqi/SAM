# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

try:
    from scipy.ndimage import binary_erosion
except ImportError:
    pass  # Don't fail if scipy is not installed. It's only necessary for this one file.


def generate_colormap(N: int, seed: int = 0) -> List[Tuple[float, float, float]]:
    """Generates a equidistant colormap with N elements."""
    random.seed(seed)

    def generate_color():
        return (random.random(), random.random(), random.random())

    return [generate_color() for _ in range(N)]


def make_visualization(
    img: torch.Tensor, source: torch.Tensor, patch_size: int = 16, class_token: bool = True
) -> Image:
    """
    Create a visualization like in the paper.

    Args:
     -

    Returns:
     - A PIL image the same size as the input.
    """

    img = np.array(img) / 255.0
    source = source.detach().cpu()

    h, w, _ = img.shape
    ph = h // patch_size
    pw = w // patch_size

    if class_token:
        source = source[:, :, 1:]

    vis = source.argmax(dim=1)
    num_groups = vis.max().item() + 1

    cmap = generate_colormap(num_groups)
    vis_img = 0

    for i in range(num_groups):
        mask = (vis == i).float().view(1, 1, ph, pw)
        mask = F.interpolate(mask, size=(h, w), mode="nearest")
        mask = mask.view(h, w, 1).numpy()

        color = (mask * img).sum(axis=(0, 1)) / mask.sum()
        mask_eroded = binary_erosion(mask[..., 0])[..., None]
        mask_edge = mask - mask_eroded

        if not np.isfinite(color).all():
            color = np.zeros(3)

        vis_img = vis_img + mask_eroded * color.reshape(1, 1, 3)
        vis_img = vis_img + mask_edge * np.array(cmap[i]).reshape(1, 1, 3)

    # Convert back into a PIL image
    vis_img = Image.fromarray(np.uint8(vis_img * 255))
    plt.imshow(vis_img)
    plt.axis('off')  # 不显示坐标轴
    plt.savefig('figures/merge.png')
    plt.close()

    return vis_img

def visualize_cosine_similarity(matrix, H, W):
    # 确保输入的H和W是有效的
    if H < 0 or H >= 64 or W < 0 or W >= 64:
        raise ValueError("Invalid H or W coordinates. They must be between 0 and 63.")

    # 计算索引
    index = H * 64 + W

    # 提取与特定token相关的相似度
    similarities = matrix[index].reshape(64, 64)

    # 可视化
    plt.imshow(similarities, cmap='hot', interpolation='nearest', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title(f"Cosine Similarities for token at ({H}, {W})")
    plt.savefig(f'figures/H={H}_W={W}_cosine.png')
    plt.close()

def compute_cosine_similarity(features, chunk_size=512):
    # features = features.view(features.size(1), -1).T  # 转换为 [h*w, dim]
    bs, n, dim = features.shape
    
    cosine_sim = torch.zeros(bs, n, n, device=features.device)

    for b in range(bs):
        for i in range(0, n, chunk_size):
            end_i = min(i + chunk_size, n)
            for j in range(0, n, chunk_size):
                end_j = min(j + chunk_size, n)
                # 计算 A 和 B 之间的余弦相似度
                cosine_sim[b, i:end_i, j:end_j] = F.cosine_similarity(
                    features[b, i:end_i].unsqueeze(1),
                    features[b, j:end_j].unsqueeze(0),
                    dim=2
                )
    return cosine_sim

def plot_similarity_matrix(matrix):
    plt.imshow(matrix, cmap='hot')
    plt.colorbar()
    plt.title("Cosine Similarity Matrix")
    plt.savefig('figures/cosine_feature.png')
    plt.close()

# 假设self.features是你的特征图，形状为[1, 256, 64, 64]
# self.features = ...