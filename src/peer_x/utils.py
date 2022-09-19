import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize
import torchvision.transforms.functional as F


def pairwise_iou(input_arr):
    """Pair-wise IOU of row-vectors of X.

    #TODO
    Args:
        input_arr:

    Returns:

    """
    intersection = input_arr[:, None, :] * input_arr[None, :, :]
    union = input_arr[:, None, :] + input_arr[None, :, :]
    result = intersection.sum(-1)/((union.sum(-1)-intersection.sum(-1)) + np.finfo(float).eps)

    return result


def to_full_mask(bbox, masks, image_size=(1000, 1000)):
    """Convert masks into full image segmentation maps.

    #TODO
    Args:
        bbox:
        masks:
        image_size:

    Returns:

    """
    zip_bb_and_masks = zip(bbox, masks)
    new_mask = np.zeros((len(bbox), image_size[0], image_size[1]), dtype=bool)
    new_mask_proba = np.zeros_like(new_mask)
    for k, v in enumerate(zip_bb_and_masks):
        x1, y1, x2, y2 = v[0][0], v[0][1], v[0][2], v[0][3]
        mask = v[1]
        img_height = y2 - y1
        img_width = x2 - x1
        mask_proba = resize(mask, (img_height, img_width))
        new_mask_proba[k, y1:y2, x1:x2] = mask_proba
        new_mask[k, y1:y2, x1:x2] = mask_proba >= .5

    return new_mask, new_mask_proba


def get_segmap_intersection(full_mask):
    """Get segmentation intersection of overlapping segmentation maps.

    #TODO
    Args:
        full_mask:

    Returns:

    """
    cur_seg_map = full_mask.copy()
    num_objects = cur_seg_map.shape[0]

    intersection_and_segmap_idx = []
    # Pair-wise comparison of segmentation maps
    for i in range(num_objects):
        for j in range(i, num_objects):
            if i != j:
                vec_i = cur_seg_map[i, :, :]
                vec_j = cur_seg_map[j, :, :]

                intersection = vec_i * vec_j
                intersection_sum = intersection.sum()
                if intersection_sum != 0:
                    intersection_and_segmap_idx.append((i, j, intersection))

    return intersection_and_segmap_idx


def remove_overlaps(intersections, full_mask_proba, full_mask):
    """Remove intersections using model confidence.

    #TODO
    Args:
        intersections:
        full_mask_proba:
        full_mask:

    Returns:

    """
    for i, j, overlaps in intersections:
        segmap_proba_i = full_mask_proba[i][overlaps]
        segmap_proba_j = full_mask_proba[j][overlaps]
        segmap_i = full_mask[i]
        segmap_j = full_mask[j]

        if segmap_proba_i.sum() > segmap_proba_j.sum():
            segmap_j[overlaps] = False
            full_mask[j] = segmap_j
        else:
            segmap_i[overlaps] = False
            full_mask[i] = segmap_i
    return full_mask


# https://pytorch.org/vision/stable/auto_examples/plot_visualization_utils.html# \
# sphx-glr-auto-examples-plot-visualization-utils-py
def show(imgs):
    """
    # TODO
    Args:
        imgs:

    Returns:

    """
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(10, 10))
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
