import numpy as np


def cal_change_size(szs, kernel, stride, padding):
    return (szs + 2 * padding - kernel) // stride + 1


def linear_size(input, kernel_size, stride_size, padding_size):
    length = len(kernel_size)
    if len(kernel_size) != len(stride_size):
        raise ValueError(
            f"The length kernel size ({len(kernel_size)}) must be same to length stride size ({len(stride_size)})"
        )
    if len(kernel_size) != len(padding_size):
        raise ValueError(
            f"The length kernel size ({len(kernel_size)}) must be same to length padding size ({len(padding_size)})"
        )
    cal = 0
    szsx, szsy = (input.shape[-2], input.shape[-1])

    while length != cal:
        szsx = cal_change_size(
            szsx, kernel_size[cal], stride_size[cal], padding_size[cal]
        )
        szsy = cal_change_size(
            szsy, kernel_size[cal], stride_size[cal], padding_size[cal]
        )
        cal += 1

    return (szsx, szsy)


def linear_size_inputsize(input_size, kernel_size, stride_size, padding_size):
    length = len(kernel_size)
    if len(kernel_size) != len(stride_size):
        raise ValueError(
            f"The length kernel size ({len(kernel_size)}) must be same to length stride size ({len(stride_size)})"
        )
    if len(kernel_size) != len(padding_size):
        raise ValueError(
            f"The length kernel size ({len(kernel_size)}) must be same to length padding size ({len(padding_size)})"
        )
    cal = 0
    szsx, szsy = input_size

    while length != cal:
        szsx = cal_change_size(
            szsx,
            kernel_size[cal]
            if not isinstance(kernel_size[cal], int)
            else kernel_size[cal][0],
            stride_size[cal]
            if not isinstance(stride_size[cal], int)
            else stride_size[cal][0],
            padding_size[cal]
            if not isinstance(padding_size[cal], int)
            else padding_size[cal][0],
        )
        szsy = cal_change_size(
            szsy,
            kernel_size[cal]
            if not isinstance(kernel_size[cal], int)
            else kernel_size[cal][1],
            stride_size[cal]
            if not isinstance(stride_size[cal], int)
            else stride_size[cal][1],
            padding_size[cal]
            if not isinstance(padding_size[cal], int)
            else padding_size[cal][1],
            # kernel_size[cal][1 if len(kernel_size[cal]) == 2 else 0],
            # stride_size[cal][1 if len(stride_size[cal]) == 2 else 0],
            # padding_size[cal][1 if len(padding_size[cal]) == 2 else 0],
        )
        cal += 1

    return szsx * szsy
