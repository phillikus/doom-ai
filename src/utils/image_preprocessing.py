from scipy.misc import imresize
import numpy as np


def scale(screen_buffer, width=None, height=None, gray=False):
    processed_buffer = screen_buffer
    if gray:
        processed_buffer = screen_buffer.astype(np.float32).mean(axis=0)

    if width is not None and height is not None:
        return imresize(processed_buffer, (height, width))
    return processed_buffer
