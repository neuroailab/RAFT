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

def plot_image_pred_gt_segments(data,
                                cmap='twilight',
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

    fig, axes = plt.subplots(1,3,figsize=figsize)
    plots = [img, pred, gt]
    titles = ['image', 'pred', 'gt']

    for i,ax in enumerate(axes):
        if cmap in cmo.cmap_d.keys():
            cmap = cmo.cmap_d[cmap]
        ax.imshow(plots[i], cmap=cmap, vmin=0, vmax=(plots[i].max()+1))
        if show_titles:
            ax.set_title(titles[i])
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', format='svg', transparent=True)
    plt.show()
