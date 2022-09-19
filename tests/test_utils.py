import numpy as np
from skimage.transform import resize
from peer_x import utils


def test_pairwise_iou():
    """Pair-wise IOU of row-vectors of X.

    #TODO
    Args:
        input_arr:

    Returns:

    """
    input_array = np.zeros((3, 6), dtype=np.float32)
    input_array[0, :] = 1
    input_array[1, 1] = 1
    input_array[2, :] = 1

    expected_output_ = [[1., 0.16666667, 1.],
                        [0.16666667, 1., 0.16666667],
                        [1., 0.16666667, 1.]]
    expected_output_ = np.array(expected_output_, dtype=np.float32)

    output_ = utils.pairwise_iou(input_array)

    assert np.all(expected_output_ == output_)
    assert np.all(expected_output_ == output_.T)


def test_to_full_mask():
    assert False


def test_get_segmap_intersection():
    assert False


def test_remove_overlaps():
    assert False
