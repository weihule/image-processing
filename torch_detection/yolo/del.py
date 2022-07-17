import os
import numpy as np
import torch


def test():
    # shifts_x shape:[w],shifts_x shape:[h]
    shifts_x = (np.arange(0, 13))
    shifts_y = (np.arange(0, 13))

    # shifts shape:[w,h,2] -> [w,h,1,2] -> [w,h,3,2] -> [h,w,3,2]
    shifts = np.array([[[shift_x, shift_y] for shift_y in shifts_y]
                        for shift_x in shifts_x],
                        dtype=np.float32)
    print(shifts)
    print(shifts.shape)

if __name__ == "__main__":
    x = np.arange(0, 5)
    y = np.arange(0, 4)
    # shift_x, shift_y = np.meshgrid(x, y)
    # print(shift_x, shift_y)

    test()
