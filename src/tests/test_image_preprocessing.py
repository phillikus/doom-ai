import pytest
from image_preprocessing import scale


@pytest.fixture
def rgb_image3x3():
    return \
        [
            [
                [10, 20, 30],
                [5, 15, 25],
                [1, 1, 1]
            ],
            [
                [10, 20, 30],
                [0, 0, 0],
                [5, 5, 5]
            ],
            [
                [0, 0, 0],
                [10, 10, 10],
                [1, 1, 1]
            ]
        ]


def test_do_not_scale_with_default_parameters(rgb_image3x3):
    scaled = scale(rgb_image3x3)

    assert len(scaled) == 3
    assert scaled[0][0] == [10, 20, 30]
    assert scaled[1][0] == [10, 20, 30]
    assert scaled[2][0] == [0, 0, 0]