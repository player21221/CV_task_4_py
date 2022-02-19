# # This is a sample Python script.
#
# # Press Shift+F10 to execute it or replace it with your code.
# # Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
#
#
# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')
#
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/


import numpy as np
import cv2


def make_gaussian_pyramide(im, nlevels=-1):
    pyr = [im.copy()]
    nlevels -= 1
    while nlevels != 0:
        tmp = cv2.pyrDown(pyr[-1])
        pyr += [tmp.copy()]
        if min(*tmp.shape) <= 2:
             break
        nlevels -= 1
    return pyr


def pyrup(im, dsize):
    dst = np.zeros(dsize[::-1], np.uint8)
    dst [::2,::2] = im
    k = np.array([1, 4, 6, 4, 1], dtype=np.float32) * 2 / 16
    dst = cv2.sepFilter2D(dst, -1, k, k)
    return dst


def make_laplacian_pyramide(im, nlevels=-1):
    im1 = im.astype(np.int16)
    pyr = []
    nlevels -= 1
    while min(*im1.shape) > 2 and nlevels != 0:
        im2 = cv2.pyrDown(im1)
        im3 = cv2.pyrUp(im2, dstsize=im1.shape[::-1])
        layer = im1 - im3
        pyr += [layer]
        im1 = im2
        nlevels -= 1

    pyr += [im1]
    return pyr


def reconstruct_laplacian_pyramide(pyr):
    im = pyr[-1]

    for layer in pyr[-2::-1]:
        im = cv2.pyrUp(im, dstsize=layer.shape[::-1])
        im += layer

    im = np.clip(im, 0, 255).astype(np.uint8)
    return im


def test_pyramidal_merge_pair():
    im1 = cv2.imread('data/merge-2/pig.png')
    im2 = cv2.imread('data/merge-2/me.png')
    m = cv2.imread('data/merge-2/mask.png', 0)
    nlevels = -1
    res = np.zeros(im1.shape, dtype=np.uint8)
    res[:, :, 0] = pyramidal_merge_pair(im1[:, :, 0], im2[:, :, 0], m, nlevels)
    res[:, :, 1] = pyramidal_merge_pair(im1[:, :, 1], im2[:, :, 1], m, nlevels)
    res[:, :, 2] = pyramidal_merge_pair(im1[:, :, 2], im2[:, :, 2], m, nlevels)
    cv2.imwrite('data/merge-2/merged.png', res)


def pyramidal_merge_pair(im1, im2, mask, nlevels=-1):
    pyrm = make_gaussian_pyramide(mask, nlevels)
    pyr1 = make_laplacian_pyramide(im1, nlevels)
    pyr2 = make_laplacian_pyramide(im2, nlevels)

    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    # layer 0
    u1 = pyr1[-1].astype(np.int32)
    u2 = pyr2[-1].astype(np.int32)
    m = pyrm[-1]

    u = (u1 * (255 - m) + u2 * m) // 255
    u = np.clip(u, -16536, 16535).astype(np.int16)

    # rest layers
    for lap1, lap2, m in zip(pyr1[-2::-1], pyr2[-2::-1], pyrm[-2::-1]):
        u = cv2.pyrUp(u, dstsize=m.shape[::-1])
        lap1 = lap1.astype(np.int32)
        lap2 = lap2.astype(np.int32)
        lap = (lap1 * (255 - m) + lap2 * m) // 255
        u += lap

    u = np.clip(u, 0, 255).astype(np.uint8)

    return u

