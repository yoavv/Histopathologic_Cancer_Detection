import cv2
import numpy as np

np.random.seed(0)

brightness = -8
saturation = 0.85
contrast = 0.7
gauss_noise = 8
channel_shift = 10
contrast_and_blur = (0.8, 3)
contrast_and_brightness = (0.7, 10)


def blur(img, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
    img_trans = cv2.filter2D(img, -1, kernel)

    return img_trans


def brightness(img, value=brightness):
    img = img.astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    value_img = hsv[:, :, 2]
    value_img = cv2.add(value_img, value)
    hsv[:, :, 2] = value_img
    img_trans = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    img_trans = cv2.convertScaleAbs(img_trans)

    return img_trans


def increase_saturation(img, th=saturation):
    img = img.astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat_img = hsv[:, :, 1]
    sat_img **= th
    hsv[:, :, 1] = sat_img
    img_trans = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    img_trans = cv2.convertScaleAbs(img_trans)

    return img_trans


def motion_blur(img, kernel_size=contrast_and_blur[1]):
    kernel_motion_blur = np.zeros((kernel_size, kernel_size))
    kernel_motion_blur[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel_motion_blur = kernel_motion_blur / kernel_size

    img_trans = cv2.filter2D(img, -1, kernel_motion_blur)

    return img_trans


def adjust_contrast(img, contrast_alpha=contrast):
    img = img.astype(np.float32)
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    original_avg = int(np.average(imghsv[:, :, 2]))
    imghsv[:, :, 2] = cv2.multiply(imghsv[:, :, 2], contrast_alpha)
    new_avg = int(np.average(imghsv[:, :, 2]))

    imghsv[:, :, 2] = cv2.add(imghsv[:, :, 2], (original_avg - new_avg) // 2)

    img = cv2.cvtColor(imghsv, cv2.COLOR_HSV2BGR)

    img = cv2.convertScaleAbs(img)

    return img


def gauss_noise(img, sigma=gauss_noise):
    img = img.astype(np.float32)
    (row, col, ch) = img.shape
    mean = 0
    sigma = sigma
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.astype(np.float32)
    gauss = gauss.reshape(row, col, ch)
    img = cv2.add(img, gauss)

    img = cv2.convertScaleAbs(img)

    return img


def channel_shift(img, intensity=channel_shift):
    img = np.rollaxis(img, 2, 0)
    min_val = np.min(img)
    max_val = np.max(img)

    channel_images = []
    for channel in img:
        channel_images.append(np.clip(channel + np.random.uniform(-intensity, intensity), min_val, max_val))

    img = np.stack(channel_images, axis=0)
    img = np.rollaxis(img, 0, 3)

    return img


def contrast_and_blur(img, contrast_alpha=contrast_and_blur[0], kernel_size=contrast_and_blur[1]):
    img = adjust_contrast(img, contrast_alpha)
    img = blur(img, kernel_size)

    return img


def contrast_and_brightness(img, contrast_alpha=contrast_and_brightness[0], value=contrast_and_brightness[1]):
    img = adjust_contrast(img, contrast_alpha)
    img = brightness(img, value)

    return img


def brightness_and_blur(img, value=contrast_and_brightness[0], kernel_size=contrast_and_brightness[1]):
    img = brightness(img, value)
    img = blur(img, kernel_size)

    return img


def brightness_and_gauss_noise(img, value=brightness, sigma=gauss_noise):
    img = brightness(img, value)
    img = gauss_noise(img, sigma)

    return img


def flip(img):
    img_trans = cv2.flip(img, flipCode=1)

    return img_trans


def grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return gray
