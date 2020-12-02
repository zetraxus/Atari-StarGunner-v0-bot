import cv2


def preprocess_image(img, target_width, target_height):
    # Convert RGB to BGR for cv2
    img = img[:, :, ::-1]
    # todo - firstly remove boundary
    downsample_img = cv2.resize(img, (target_width, target_height))
    greyscale_downsample_img = cv2.cvtColor(downsample_img, cv2.COLOR_RGB2GRAY)
    return greyscale_downsample_img
