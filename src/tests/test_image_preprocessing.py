import pytest
from image_preprocessing import scale
import numpy as np

@pytest.fixture
def rgb_image4x4():
    return np.array(\
        [
            [
                [10, 10, 10, 10],
                [10, 10, 10, 10],
                [10, 10, 10, 10],
                [10, 10, 10, 10]
            ],
            [
                [20, 20, 20, 20],
                [20, 20, 20, 20],
                [20, 20, 20, 20],
                [20, 20, 20, 20]
            ],
            [
                [10, 10, 10, 10],
                [10, 10, 10, 10],
                [10, 10, 10, 10],
                [10, 10, 10, 10]
            ]
        ])


def test_do_not_scale_with_default_parameters(rgb_image4x4):
    scaled = scale(rgb_image4x4)

    assert len(scaled) == 3
    assert np.array_equal(scaled[0][0], [10, 10, 10, 10])
    assert np.array_equal(scaled[1][0], [20, 20, 20, 20])
    assert np.array_equal(scaled[2][0], [10, 10, 10, 10])


def test_scale_2x2(rgb_image4x4):
    scaled = scale(rgb_image4x4, 2, 2)

    assert len(scaled) == 3
    assert scaled[0][0] == [10, 20, 30, 40]
    assert scaled[1][0] == [10, 20, 30, 40]
    assert scaled[2][0] == [0, 0, 0, 0]
    assert scaled[3][0] == [1, 2, 3, 4]
