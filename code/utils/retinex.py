import numpy as np
import cv2

def singleScaleRetinex(img, sigma):

    img[img==0] = 1
    retinex = np.log10(img) - np.expand_dims(np.log10(cv2.GaussianBlur(img, (0, 0), sigma)),-1)

    return retinex


def multiScaleRetinex(img, sigma_list):
    retinex = np.zeros_like(img, dtype='float')
    for sigma in sigma_list:
        retinex += singleScaleRetinex(img, sigma)
    retinex = retinex / len(sigma_list)

    # retinex = retinex + np.amin(retinex)
    # retinex = retinex / np.amax(retinex)
    # retinex = retinex * 255.0
    # retinex = retinex.astype('uint8')

    return retinex


def colorRestoration(img, alpha, beta):
    img_sum = np.sum(img, axis=2, keepdims=True)

    color_restoration = beta * (np.log10(alpha * img) - np.log10(img_sum))

    return color_restoration


def simplestColorBalance(img, low_clip, high_clip):
    total = img.shape[0] * img.shape[1]
    for i in range(img.shape[2]):
        unique, counts = np.unique(img[:, :, i], return_counts=True)
        current = 0
        for u, c in zip(unique, counts):
            if float(current) / total < low_clip:
                low_val = u
            if float(current) / total < high_clip:
                high_val = u
            current += c

        img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)

    return img


def MSRCR(img, sigma_list, G, b, alpha, beta, low_clip, high_clip):
    img = np.float64(img) + 1.0

    img_retinex = multiScaleRetinex(img, sigma_list)
    img_color = colorRestoration(img, alpha, beta)
    img_msrcr = G * (img_retinex * img_color + b)

    for i in range(img_msrcr.shape[2]):
        img_msrcr[:, :, i] = (img_msrcr[:, :, i] - np.min(img_msrcr[:, :, i])) / \
                             (np.max(img_msrcr[:, :, i]) - np.min(img_msrcr[:, :, i])) * \
                             255

    img_msrcr = np.uint8(np.minimum(np.maximum(img_msrcr, 0), 255))
    img_msrcr = simplestColorBalance(img_msrcr, low_clip, high_clip)

    return img_msrcr


def automatedMSRCR(img, sigma_list):
    img = np.float64(img) + 1.0

    img_retinex = multiScaleRetinex(img, sigma_list)

    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break

        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.1:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.1:
                high_val = u / 100.0
                break

        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)

        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                               * 255

    img_retinex = np.uint8(img_retinex)

    return img_retinex


def MSRCP(img, sigma_list, low_clip, high_clip):
    img = np.float64(img) + 1.0

    intensity = np.sum(img, axis=2) / img.shape[2]

    retinex = multiScaleRetinex(intensity, sigma_list)

    intensity = np.expand_dims(intensity, 2)
    retinex = np.expand_dims(retinex, 2)

    intensity1 = simplestColorBalance(retinex, low_clip, high_clip)

    intensity1 = (intensity1 - np.min(intensity1)) / \
                 (np.max(intensity1) - np.min(intensity1)) * \
                 255.0 + 1.0

    img_msrcp = np.zeros_like(img)

    for y in range(img_msrcp.shape[0]):
        for x in range(img_msrcp.shape[1]):
            B = np.max(img[y, x])
            A = np.minimum(256.0 / B, intensity1[y, x, 0] / intensity[y, x, 0])
            img_msrcp[y, x, 0] = A * img[y, x, 0]
            img_msrcp[y, x, 1] = A * img[y, x, 1]
            img_msrcp[y, x, 2] = A * img[y, x, 2]

    img_msrcp = np.uint8(img_msrcp - 1.0)

    return img_msrcp

# img = cv2.imread("fake/0000.png")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = cv2.resize(img, (299, 299))

if __name__ == '__main__':
    images = ['fake/0000.png', 'fake/0.png', 'fake/0001.png', 'fake/0001_00_00_01_0.jpg',
              'real/0000.png', 'real/0.png', 'real/0001.png', 'real/0001_00_00_01_0.jpg']

    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(8, 8))
    columns = 4
    rows = 4

    for index, img in enumerate(images):
        img = cv2.imread(img)
        img = cv2.resize(img, (299, 299))

        fig.add_subplot(4, 4, 2 * (index) + 1)
        tmp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(tmp)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, -1)
        new_img = automatedMSRCR(img, [2, 4, 8])
        # new_img = multiScaleRetinex(img, [5, 10, 15])
        # print(new_img.shape)
        # new_img = cv2.cvtColor(new_img[:,:,0], cv2.COLOR_GRAY2RGB)
        fig.add_subplot(4, 4, 2 * (index) + 2)
        plt.imshow(new_img[:,:,0], cmap='gray')
    plt.show()

# img = multiScaleRetinex(img, [10, 20, 30])
# img = automatedMSRCR(img, [10,20,30])
# print(img)
# print(img.shape)
# cv2.imshow("img", img)
# cv2.waitKey(0)