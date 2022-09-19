import logging
import time

import hydra
import numpy as np
from glob import glob
import pathlib
import os

import skimage
from omegaconf import DictConfig
from tqdm import tqdm

from peer_x import utils
log = logging.getLogger(__name__)


@hydra.main(config_path="../configs/post_processing/", config_name="default.yaml")
def main(config: DictConfig) -> None:
    logging.info(f"Starting post-processing pipeline")
    start_time = time.time()
    image_dir = config.glob_image_dir
    model_output_dir = config.glob_model_output_dir
    processed_output_dir = config.processed_output_dir

    if not os.path.exists(processed_output_dir):
        os.makedirs(processed_output_dir)

    image_paths = glob(image_dir)
    image_paths = np.sort(image_paths)

    output_paths = glob(model_output_dir)
    output_paths = np.sort(output_paths)

    # TODO refactor consider parallel processing
    for idx, val in enumerate(tqdm(output_paths)):
        logging.info(f"Processing {val}")
        cur_output = np.load(val)
        cur_image = skimage.io.imread(image_paths[idx])
        cur_image_shape = cur_image.shape[:-1]

        # Convert mask to full mask
        new_mask, new_mask_proba = utils.to_full_mask(cur_output["boxes"], cur_output["masks"], cur_image_shape)

        # Remove overlapping segmentations
        intersections = utils.get_segmap_intersection(new_mask)
        new_full_mask = utils.remove_overlaps(intersections, new_mask_proba, new_mask)

        #
        cur_output = dict(cur_output)
        cur_output["masks"] = new_full_mask

        # Save as new npz file
        save_path = os.path.join(processed_output_dir, pathlib.Path(val).name)
        logging.info(f"Saving {save_path}")
        np.savez(save_path, **cur_output)

    log.info(f"Done!")
    log.info(f"{time.time() - start_time} seconds")


if __name__ == "__main__":
    main()
