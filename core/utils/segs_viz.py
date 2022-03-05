import numpy as np
import torch
import matplotlib.pyplot as plt
import cmocean.cm as cmo

def tensor_to_arr(tensor, ex=0):
    if len(tensor.shape) == 4:
        tensor = tensor[ex]
    return tensor.detach().permute(1, 2, 0).cpu().numpy()

def viz(tensor, ex=0):
    im = tensor_to_arr(tensor, ex)
    if im.max() > 2.0:
        im = im / 255.0
    plt.imshow(im)

from PIL import ImageColor
import json
from pathlib import Path

def get_palette(colors_json='./colors.json', i=0):
    colors = json.loads(Path(colors_json).read_text(encoding='utf-8'))
    colors_hex = colors[i]
    colors_rgb = [ImageColor.getcolor(col, "RGB") for col in colors_hex]
    return colors_rgb

def plot_palette(i=3):
    colors = get_palette(i=i)
    arr = np.zeros((20,50,3))
    for i,c in enumerate(colors):
        arr[:,10*i:10*(i+1),:] = np.stack([np.array(c)]*10, 0) / 255.
    plt.imshow(arr)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def seg_to_rgb(seg, colors):
    size = seg.shape[:2]
    rgb = np.zeros((size[0], size[1], 3))
    for i,c in enumerate(colors):
        rgb[(seg == i),:] = c
    return rgb / 255.

def plot_image_pred_gt_segments(data,
                                cmap='twilight',
                                bg_color=(0,0,0),
                                figsize=(12,4),
                                show_titles=False,
                                save_path=None):

    assert all((k in data.keys() for k in ('image', 'pred_segments', 'gt_segments')))

    img = data['image'].permute(1,2,0).numpy()
    size = img.shape[0:2]

    gt = np.zeros(size)
    pred = np.zeros(size)

    N, _N = len(data['pred_segments']), len(data['gt_segments'])
    assert N == _N

    for n in range(N):
        gt += data['gt_segments'][n].astype(gt.dtype) * (n+1)
        pred += data['pred_segments'][n].astype(pred.dtype) * (n+1)

    if isinstance(cmap, int):
        colors = get_palette(i=cmap)
        colors.insert(0, bg_color)
        gt, pred = seg_to_rgb(gt, colors), seg_to_rgb(pred, colors)

    fig, axes = plt.subplots(1,3,figsize=figsize)
    plots = [img, pred, gt]
    titles = ['image', 'pred', 'gt']

    for i,ax in enumerate(axes):
        if cmap in cmo.cmap_d.keys():
            cmap = cmo.cmap_d[cmap]
        if isinstance(cmap, int):
            ax.imshow(plots[i])
        else:
            ax.imshow(plots[i], cmap=cmap, vmin=0, vmax=(plots[i].max()+1))
        if show_titles:
            ax.set_title(titles[i])
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', format='svg', transparent=True)
    plt.show()
