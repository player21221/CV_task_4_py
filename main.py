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


def pyrUp(im, dsize):
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

def test_pyramidal_merge_pair(im1, im2, m, nlevels = -1):
    # im1 = cv2.imread('data/merge-2/pig.png')
    # im2 = cv2.imread('data/merge-2/me.png')
    # m = cv2.imread('data/merge-2/mask.png', 0)
    # nlevels = -1
    res = np.zeros(im1.shape, dtype=np.uint8)
    res[:, :, 0] = pyramidal_merge_pair(im1[:, :, 0], im2[:, :, 0], m, nlevels)
    res[:, :, 1] = pyramidal_merge_pair(im1[:, :, 1], im2[:, :, 1], m, nlevels)
    res[:, :, 2] = pyramidal_merge_pair(im1[:, :, 2], im2[:, :, 2], m, nlevels)
    #cv2.imwrite('data/merge-2/merged.png', res)
    return res

def define_mask(im1, im2, trh = 128):
    # mask=np.zeros(im1.shape, dtype=np.uint8)
    # for i in range(0, im1.shape[0]):
    #     for j in range(0, im2.shape[1]):
    #         if (im1[i,j] < im2[i,j]):
    #             mask[i,j]=1
    # return mask

    return (abs(im1.astype(np.int)-trh) < abs(im2.astype(np.int)-trh))

def define_mask_for_color(imm, headroom=5):
    """the darkest image must be first"""
    # imm = [np.array(pic1(height, wight, color)),np.array(pic2), ...]
    # max = np.max(imm, axis=3)
    # bool = np.max(imm, axis=3)<(255-headroom)
    # maxbool = np.max(imm, axis=3)*(np.max(imm, axis=3)<(255-headroom))
    # returnage = np.argmax(np.max(imm, axis=3) * (np.max(imm, axis=3) < (255 - headroom)), axis=0)
    # for i in range(0, len(imm)):
    #     # cv2.imshow("bool "+ str(i),bool[i].astype(np.uint8) * 255)
    #     # cv2.imshow("img " + str(i),imm[i].astype(np.uint8))
    #     # cv2.imshow("max " + str(i), max[i].astype(np.uint8))
    #     cv2.imshow("maxbool " + str(i), maxbool[i].astype(np.uint8))
    # cv2.imshow("returnage ", returnage.astype(np.uint8) * 128)
    # cv2.waitKey()
    return np.argmax(np.max(imm, axis=3)*(np.max(imm, axis=3)<(255-headroom)),axis=0)



def define_multi_mask(im, trh = 128):
    return np.argmin(trh-im, axis=2)

def sum_multi_mask(mask, im):
    lapl = np.zeros(mask.shape[0:1])
    for i in range(0,mask.shape[2]):
        lapl += im[:,:,i]*((i-mask) == 0)
    # lapl = sum(im[:,:,i]*((i-mask) == 0),)
    return lapl

def pyramidal_merge_multiframe(imm, mask, nlevels=-1,colors=3):
    # maskm = [np.zeros(mask.shape[1:2]) for _ in range(len(imm))]
    mpyrm = [0]*len(imm)
    ipyrm = [[0]*colors for _ in range(len(imm))]
    for i in range(0,len(imm)):
        masktmp= (mask == i).astype(np.int16)*255
        # mpyrm[i] = make_gaussian_pyramide(maskm[i],nlevels)
        # ipyrm[i] = make_laplacian_pyramide(imm[i],nlevels)
        mpyrm[i] = make_gaussian_pyramide(masktmp,nlevels)
        for col in range(colors):
            ipyrm[i][col] = make_laplacian_pyramide(imm[i][:,:,col],nlevels)
    # pyrm = make_gaussian_pyramide(mask, nlevels)
    # pyr1 = make_laplacian_pyramide(im1, nlevels)
    # pyr2 = make_laplacian_pyramide(im2, nlevels)

    # mask = cv2.GaussianBlur(mask, (7, 7), 0)

        # layer 0
    u = [np.zeros(ipyrm[0][0][-1].shape) for _ in range(colors)]
    for col in range(colors):

        for i in range (len(imm)):
            u[col]+=ipyrm[i][col][-1]*mpyrm[i][-1]


        # u1 = pyr1[-1].astype(np.int32)
        # u2 = pyr2[-1].astype(np.int32)
        # m = pyrm[-1]
        #
        # u = (u1 * (255 - m) + u2 * m) // 255
        # u = np.clip(u, -16536, 16535).astype(np.int16)

        # for ipyri, mpyri in zip(ipyrm[:][-2::-1],mpyrm[:][-2::-1]):
        #     u = cv2.pyrUp(u, dstsize=mpyri[0].shape[::-1])
        #     for i in range(len(imm)):
        #         u += ipyri[i] * mpyri[i]

        for i in range(len(ipyrm[0][col])-2,-1,-1):
            u[col] = cv2.pyrUp(u[col], dstsize=mpyrm[0][i].shape[::-1])
            for j in range(len(imm)):
                u[col] += ipyrm[j][col][i] * mpyrm[j][i]

        # rest layers
        # for lap1, lap2, m in zip(pyr1[-2::-1], pyr2[-2::-1], pyrm[-2::-1]):
        #     u = cv2.pyrUp(u, dstsize=m.shape[::-1])
        #     lap1 = lap1.astype(np.int32)
        #     lap2 = lap2.astype(np.int32)
        #     lap = (lap1 * (255 - m) + lap2 * m) // 255
        #     u += lap
    u = np.array(u)
    u = np.moveaxis(u, 0, -1) / 255

    return u

def post_process(u, normalize_histogram=0, brgcnt=[0,1], brcn_hsv=[0,1],B = 0):
     #2 for hsv, 0 for ycc
    if normalize_histogram == 1:  #only for hsv
        #https://docs.opencv.org/3.4/d5/daf/tutorial_py_histogram_equalization.html
        # create a CLAHE object (Arguments are optional).
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        utmp=u[:,:,B].astype(np.uint8)
        # u[:,:,B] = clahe.apply(u[:,:,B])
        utmp = clahe.apply(utmp)
        # u[:,:,B]=utmp
    elif normalize_histogram == 2:
        u[:,:,B] = cv2.equalizeHist(u[:,:,B])
    u[:,:,B]=u[:,:,B]*brcn_hsv[1]+brcn_hsv[0]
    u = np.clip((u*brgcnt[1]+brgcnt[0]), 0, 255).astype(np.uint8)
    return u

# def pyramidal_merge_multiframe(ims, mask, nlevels=-1):
#     pyrm = make_gaussian_pyramide(mask)


def conv_color(imm, param):
    im_hsv=[0 for _ in range(len(imm))]
    for i in range(len(imm)):
        im_hsv[i] = cv2.cvtColor(imm[i], param)
    return im_hsv

conv_bgr2hsv = lambda imm: conv_color(imm, cv2.COLOR_BGR2HSV)
conv_bgr2yuv = lambda imm: conv_color(imm, cv2.COLOR_BGR2YUV)
conv_hsv2brg = lambda imm: conv_color(imm, cv2.COLOR_HSV2BGR)
conv_yuv2brg = lambda imm: conv_color(imm, cv2.COLOR_HSV2BGR)
conv_bgr2lab = lambda imm: conv_color(imm, cv2.COLOR_BGR2LAB)
conv_lab2bgr = lambda imm: conv_color(imm, cv2.COLOR_LAB2BGR)



def test_change_val(val):
    val+=1
    return


bgr2hsv = lambda bgr: cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
bgr2yuv = lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
hsv2bgr = lambda hsv: cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
yuv2bgr = lambda img: cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
bgr2lab = lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
lab2bgr = lambda img: cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

# def bgr2val_col_rat(im):
#     res = np.empty(im.shape)
#     res[:,:,0]=(im[:,:,0]+im[:,:,1]+im[:,:,2])/3
#     res[:,:,1]=(im[:,:,0]+im[:,:,2])/(im[:,:,1]*2)
#     res[:,:,2]=(im[:,:,0])/(im[:,:,2])
#     return res
#
# def val_col_rat2bgr(im):
#     res = np.empty(im.shape)
#     res[:, :, 0] = res[:,:,0]
#     res[:, :, 1] = (im[:, :, 0] + im[:, :, 2]) / (im[:, :, 1] * 2)
#     res[:, :, 2] = (im[:, :, 0]) / (im[:, :, 2])

filenames = [
    "photo1645261254.jpeg",
    "photo1645261254(1).jpeg"
]
resultname = "result_sigma_coworking"
# filenames = [
#     "photo_2022-02-21_20-17-12.jpg",
#     "photo_2022-02-21_20-17-14.jpg",
#     "photo_2022-02-21_20-17-17.jpg",
#     "photo_2022-02-21_20-17-20.jpg",
#     "photo_2022-02-21_20-17-23.jpg",
#     "photo_2022-02-21_20-17-26.jpg",
#     "photo_2022-02-21_20-17-30.jpg",
#     "photo_2022-02-21_20-17-33.jpg",
#     "photo_2022-02-21_20-17-36.jpg",
#     "photo_2022-02-21_20-17-40.jpg"
# ]
# resultname = "result_beta_lab"

imm=[0 for _ in range(len(filenames))]
for f in range(len(filenames)):
    imm[f]=cv2.imread(filenames[f])



#im1 = cv2.imread("photo1645261254.jpeg") # m=1 gets form dark
# im2 = cv2.imread("photo1645261254(1).jpeg") # m=0 gets from bright
# # msk = define_mask(im2, im1).astype(np.uint8)*255
#
# imm=[im1, im2]

#im2 = cv2.cvtColor(im1, cv2.COLOR_BGR2HSV)

msk=define_mask_for_color(imm).astype(np.uint8)

# im1m = im1.astype(np.int) - 128
# im2m = im2.astype(np.int) - 128

#result = test_pyramidal_merge_pair(im1,im2,msk)#[:,:,1])

result = np.empty(imm[0].shape)

# for color in range(3):
#     result[:,:,color]=pyramidal_merge_multiframe(imm[:,:,:,color],msk)

#result = pyramidal_merge_multiframe(imm,msk, normalize_histogram=0,brgcnt=[-100,1])
result = pyramidal_merge_multiframe(conv_bgr2yuv(imm), msk)
brg_cont=[70, .5]
# brg_cont=[-50, .7]
result = yuv2bgr(post_process(result, normalize_histogram=0, brcn_hsv=brg_cont,B=0))

resultfilename = resultname + "_" + str(brg_cont)
cv2.imshow(resultfilename + "_mask", (msk*(255/(len(imm)-1))).astype(np.uint8))
cv2.imwrite(resultfilename + "_mask" + ".jpeg", (msk*(255/(len(imm)-1))).astype(np.uint8))
#for f in range(len(filenames)): cv2.imshow(filenames[f],imm[f])
# cv2.imshow("im1", im1)
# cv2.imshow("im2", im2)

cv2.imshow(resultfilename, result.astype(np.uint8))
cv2.imwrite(resultfilename + ".jpeg",result.astype(np.uint8))
# cv2.imshow("im1-128", abs(im1m).astype(np.uint8))
# cv2.imshow("im2-128", abs(im2m).astype(np.uint8))

cv2.waitKey()