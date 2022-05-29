import torch as th


BOX_BLUR_3X3 = th.full((3, 3), 1.0 / 9.0, dtype=th.float32)

BOX_BLUR_5X5 = th.full((5, 5), 1.0 / 25.0, dtype=th.float32)

GAUSSIAN_BLUR_3x3 = (1.0 / 16.0) * th.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=th.float32)

VERTICAL_SOBEL_3x3 = th.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=th.float32)

HORIZONTAL_SOBEL_3x3 = th.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=th.float32)


if __name__ == '__main__':

    print(BOX_BLUR_3X3)
    print(BOX_BLUR_5X5)
    print(GAUSSIAN_BLUR_3x3)
    print(VERTICAL_SOBEL_3x3)
    print(HORIZONTAL_SOBEL_3x3)
