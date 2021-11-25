import sys
import os
import math
import random
import numpy as np
# from scipy import misc
from PIL import Image, ImageEnhance
from skimage import transform
import imageio
import cv2
from scipy.ndimage.interpolation import rotate
import scipy.ndimage as nd
# from keras_preprocessing import image as Keras_Image_Proc

def normalize_pixel(x, v0, v, m, m0):
    """
    From Handbook of Fingerprint Recognition pg 133
    Normalize job used by Hong, Wan and Jain(1998)
    similar to https://pdfs.semanticscholar.org/6e86/1d0b58bdf7e2e2bb0ecbf274cee6974fe13f.pdf equation 21
    :param x: pixel value
    :param v0: desired variance
    :param v: global image variance
    :param m: global image mean
    :param m0: desired mean
    :return: normilized pixel
    """
    dev_coeff = math.sqrt((v0 * ((x - m)**2)) / v)
    return m0 + dev_coeff if x > m else m0 - dev_coeff

def get_mask_ROI(im, w, threshold=.2):
    """
    Returns mask identifying the ROI. Calculates the standard deviation in each image block and threshold the ROI
    It also normalises the intesity values of
    the image so that the ridge regions have zero mean, unit standard
    deviation.
    :param im: Image
    :param w: size of the block
    :param threshold: std threshold
    :return: segmented_image
    """
    (y, x) = im.shape
    threshold = np.std(im)*threshold

    image_variance = np.zeros(im.shape)
    segmented_image = im.copy()
    mask = np.ones_like(im)

    for i in range(0, x, w):
        for j in range(0, y, w):
            box = [i, j, min(i + w, x), min(j + w, y)]
            block_stddev = np.std(im[box[1]:box[3], box[0]:box[2]])
            image_variance[box[1]:box[3], box[0]:box[2]] = block_stddev

    # apply threshold
    mask[image_variance < threshold] = 0

    # smooth mask with a open/close morphological filter
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(w*2, w*2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    blank_mask = np.ones_like(im)

    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    cv2.fillPoly(blank_mask, [cnts], (255,255,255))
    # mask = cv2.cvtColor(blank_mask, cv2.COLOR_BGR2GRAY)

    # normalize segmented image
    # segmented_image *= mask
    # im = normalise(im)
    # mean_val = np.mean(im[mask==0])
    # std_val = np.std(im[mask==0])
    # norm_img = (im - mean_val)/(std_val)

    return mask


def contrast_enhance(images):
    new_images = []
    for img in images:
        new_images.append(np.asarray(ImageEnhance.Contrast(Image.fromarray(img)).enhance(2)))
    return np.array(new_images)

def normalize(im, m0, v0):
    m = np.mean(im)
    v = np.std(im) ** 2
    (y, x) = im.shape
    normilize_image = im.copy()
    for i in range(x):
        for j in range(y):
            normilize_image[j, i] = normalize_pixel(im[j, i], v0, v, m, m0)
    return normilize_image

def segment_fp(images):
    segmented_images = []
    failed = []
    for i, image in enumerate(images):
        ret, thresh = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(64,64))
        img2 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        img2 = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img2, connectivity=4)
        try:
            max_label, max_size = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)], key=lambda x: x[1])
            img2 = np.ones(output.shape) * 255
            img2[output == max_label] = 0
        except:
            nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
            print('failed1')
            img2 = images[i]
        
        try:
            x,y,w,h = cv2.boundingRect(np.uint8(img2))
        except:
            failed.append((i, 'second'))
            print('failed2')
            segmented_images.append(images[i])
            continue
        segmented = image.copy()[y:y+h, x:x+w]
        if segmented.shape[0] == 0 or segmented.shape[1] == 0:
            failed.append((i, 'third'))
            print('failed3')
            segmented_images.append(images[i])
            continue
        segmented_images.append(segmented)
    return np.array(segmented_images)


def scale_fp(images, MIN_WIDTH=480):
    for i, image in enumerate(images):
        h, w = tuple(image.shape)
        if (h < w):
            scale_percent = math.ceil(MIN_WIDTH/image.shape[0] * 100)
        else:
            scale_percent = math.ceil(MIN_WIDTH/image.shape[1] * 100) # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        images[i] = cv2.resize(image, dim)
    return images

def random_scale(images, max_percent):
    images_new = images
    _h, _w = tuple(images[0].shape)
    scales = []
    for i in range(images_new.shape[0]):
        if random.randint(0, 1) == 1:
            percent = np.clip(random.uniform(max_percent[0], max_percent[1]), max_percent[0], max_percent[1])
            scales.append(percent)
            width = int(images[i].shape[1] * percent)
            height = int(images[i].shape[0] * percent)
            dim = (width, height)
            y1, x1 = max(0, height - _h) // 2, max(0, width - _w) // 2
            y2, x2 = y1 + _h, x1 + _w
            bbox = np.array([y1,x1,y2,x2])
            # Map back to original image coordinates
            bbox = (bbox / percent).astype(np.int)
            y1, x1, y2, x2 = bbox
            cropped_img = images[i][y1:y2, x1:x2]
            resize_height, resize_width = min(height, _h), min(width, _w)
            pad_height1, pad_width1 = (_h - resize_height) // 2, (_w - resize_width) //2
            pad_height2, pad_width2 = (_h - resize_height) - pad_height1, (_w - resize_width) - pad_width1
            pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0,0)] * (images[i].ndim - 2)
            result = cv2.resize(cropped_img, (resize_width, resize_height))
            result = np.pad(result, pad_spec, mode='constant', constant_values=255)
            images_new[i] = result
        else:
            scales.append(1.0)
    return images_new, scales

def random_contrast(images, min_contrast, max_contrast, u=0.5):
    images_new = images
    for i in range(images_new.shape[0]):
        if np.random.random() < u:
            contrast = np.clip(random.uniform(min_contrast, max_contrast), min_contrast, max_contrast)
            images_new[i] = np.asarray(ImageEnhance.Contrast(Image.fromarray(np.uint8(images[i]))).enhance(contrast))
    return images_new


def pad_image(img_data, output_width, output_height):
    height, width = img_data.shape
    output_img = np.ones((output_height, output_width), dtype=np.int32) * 255
    margin_h = (output_height - height) // 2
    margin_w = (output_width - width) // 2
    output_img[margin_h:margin_h+height, margin_w:margin_w+width] = img_data
    return output_img

# Calulate the shape for creating new array given (h,w)
def get_new_shape(images, size=None, n=None):
    shape = list(images.shape)
    if size is not None:
        h, w = tuple(size)
        shape[1] = h
        shape[2] = w
    if n is not None:
        shape[0] = n
    shape = tuple(shape)
    return shape

# def random_crop(images, size, u=0.4):
#     n, _h, _w = images.shape[:3]
#     h, w = tuple(size)
#     shape_new = get_new_shape(images, size)
#     assert (_h>=h and _w>=w)

#     images_new = np.ndarray(shape_new, dtype=images.dtype)

#     y = np.random.randint(low=0, high=_h-h+1, size=(n))
#     x = np.random.randint(low=0, high=_w-w+1, size=(n))

#     for i in range(n):
#         if np.random.random() < u:
#             images_new[i] = images[i, y[i]:y[i]+h, x[i]:x[i]+w]

#     return images_new

def random_crop(images, size, u=0.3):
     images_new = []
     min_h, min_w = tuple(size)
     for i, img in enumerate(images):
        if i%2==0:
            if np.random.random() < u:
                 _h, _w = tuple(img.shape)
                 if _h <= min_h or _w <= min_w:
                    h = _h
                    w = _w
                    x = 0
                    y = 0
                 else:
                    h = np.random.randint(low=min_h, high=_h)
                    w = np.random.randint(low=min_w, high=_w)
                    y = np.random.randint(low=0, high=_h-h+1)
                    x = np.random.randint(low=0, high=_w-w+1)
                 images_new.append(img[y:y+h, x:x+w])
            else:
                 images_new.append(img)
        else:
            images_new.append(img)
     return images_new

def center_crop_raw(images, size):
    # if len(images[0].shape) == 2:
    #   images_new = np.ndarray((len(images), size[0], size[1]), np.float32)
    # else:
    #   images_new = np.ndarray((len(images), size[0], size[1], images.shape[-1]), np.float32)
    images_new = []
    for i, image in enumerate(images):
        top_crop = False
        try:
            _h, _w = tuple(image.shape)
            if _h >= 450:
                top_crop = True
        except:
            print('FAILED: ', image)
        # print(image.shape)
        h, w = tuple(size)
        if _h < h or _w < w:
            if _h < h and _w >= w:
                dims = (h - _h, 0)
            elif _h >= h and _w < w:
                dims = (0, w - _w)
            elif _h < h and _w < w:
                dims = (h - _h, w - _w)
            image = np.squeeze(padding(image[None, :, :], dims))
            # print('PADDED {} to {}'.format(image.shape, new_image.shape))
            try:
                _h, _w = tuple(image.shape)
            except:
                print(image.shape)
        assert (_h>=h and _w>=w)

        if not top_crop:
            y = int(round(0.5 * (_h - h)))
        else:
            y = 0
        x = int(round(0.5 * (_w - w)))

        images_new.append(image[y:y+h, x:x+w])
    images_new = np.array(images_new)
    return images_new

def top_crop_raw(images, size):
    # if len(images[0].shape) == 2:
    #   images_new = np.ndarray((len(images), size[0], size[1]), np.float32)
    # else:
    #   images_new = np.ndarray((len(images), size[0], size[1], images.shape[-1]), np.float32)
    images_new = []
    for i, image in enumerate(images):
        top_crop = False
        try:
            _h, _w = tuple(image.shape)
        except:
            print('FAILED: ', image)
        # print(image.shape)
        h, w = tuple(size)
        if _h < h or _w < w:
            if _h < h and _w >= w:
                dims = (h - _h, 0)
            elif _h >= h and _w < w:
                dims = (0, w - _w)
            elif _h < h and _w < w:
                dims = (h - _h, w - _w)
            image = np.squeeze(padding(image[None, :, :], dims))
            # print('PADDED {} to {}'.format(image.shape, new_image.shape))
            try:
                _h, _w = tuple(image.shape)
            except:
                print(image.shape)
        assert (_h>=h and _w>=w)

        y = 0
        x = int(round(0.5 * (_w - w)))

        images_new.append(image[y:y+h, x:x+w])
    images_new = np.array(images_new)
    return images_new

def center_crop(images, size):
    n, _h, _w = images.shape[:3]
    h, w = tuple(size)
    assert (_h>=h and _w>=w)

    y = int(round(0.5 * (_h - h)))
    x = int(round(0.5 * (_w - w)))

    images_new = images[:, y:y+h, x:x+w]

    return images_new

def top_center_crop(images, size):
    images_new = np.ndarray((images.shape[0], size[1], size[1]), np.float32)
    h, w = tuple(size)
    for i in range(len(images)):
        _h , _w = images[i].shape[0], images[i].shape[1]
        if _h < (h + w):
            dims = ( (h+w) -_h, 0)
            images[i] = np.squeeze(padding(images[i][None], dims))
            _h, _w = tuple(images[i].shape)
        y = h
        x = int(round(0.5 * (_w - w)))
        
        images_new[i] = images[i][y:y+w, x:x+w]
    return images_new

def random_flip(images):
    images_new = images.copy()
    flips = np.random.rand(images_new.shape[0])>=0.5
    
    for i in range(images_new.shape[0]):
        if flips[i]:
            images_new[i] = np.fliplr(images[i])

    return images_new

def flip(images):
    images_new = images.copy()
    for i in range(images_new.shape[0]):
        images_new[i] = np.fliplr(images[i])

    return images_new

def resize_raw(images, size):
    n = images.shape[0]
    h, w = tuple(size)
    # shape_new = get_new_shape(images, size)

    images_new = []
    for i in range(n):
        images_new.append(transform.resize(images[i], (h,w), preserve_range=True))
        # images_new[i] = Image.resize(images[i], (h,w))
        # images_new[i] = np.resize(images[i], (h,w))
    images_new = np.array(images_new)
    return images_new

def resize(images, size):
    n, _h, _w = images.shape[:3]
    h, w = tuple(size)
    shape_new = get_new_shape(images, size)

    images_new = np.ndarray(shape_new, dtype=images.dtype)
    for i in range(n):
        # images_new[i] = transform.resize(images[i], (h,w), preserve_range=True)
        images_new[i] = cv2.resize(images[i], (h,w))
        # images_new[i] = Image.resize(images[i], (h,w))
        # images_new[i] = np.resize(images[i], (h,w))
    return images_new

def padding(images, padding):
    n, _h, _w = images.shape[:3]
    if len(padding) == 2:
        pad_t = pad_b = padding[0]
        pad_l = pad_r = padding[1]
    else:
        pad_t, pad_b, pad_l, pad_r = tuple(padding)
       
    size_new = (_h + pad_t + pad_b, _w + pad_l + pad_r)
    shape_new = get_new_shape(images, size_new)
    images_new = np.ones(shape_new, dtype=images.dtype) * 255
    images_new[:, pad_t:pad_t+_h, pad_l:pad_l+_w] = images

    return images_new

def standardize_images(images, standard):
    if standard=='mean_scale':
        mean = 127.5
        std = 128.0
    elif standard=='scale':
        mean = 0.0
        std = 255.0
    images_new = images.astype(np.float32)
    images_new = (images_new - mean) / std
    return images_new

'''def random_translate(images, max_dims):

    n, _h, _w = images.shape[:3]
    pad_x = _w + max_dims[0]
    pad_y = _h + max_dims[1]
    images_temp = padding(images, (pad_y, pad_x))
    images_new = images.copy().astype(np.float32)

    for i in range(n):
        shift_x = np.random.randint(0, max_dims[0]+1)
        shift_y = np.random.randint(0, max_dims[1]+1)
        images_new[i] = images_temp[i, pad_y+shift_y:pad_y+shift_y+_h, 
                            pad_x+shift_x:pad_x+shift_x+_w]

    return images_new'''

def random_translate(images, units):
    units_x, units_y = units
    images_new = images
    translations = []
    _h, _w = tuple(images[0].shape)
    for i in range(images_new.shape[0]):
      units_x = random.randint(0, units_x)
      units_y = random.randint(0, units_y)
      flip_x = 1 if np.random.random() < 0.5 else -1
      flip_y = 1 if np.random.random() < 0.5 else -1
      x_units = int(flip_x * units_x)
      y_units = int(flip_y * units_y)
      # units_x = random.uniform(-1.0*units_x, units_x)
      # units_y = random.uniform(-1.0*units_y, units_y)
      M = np.float32([ [1,0,x_units], [0,1,y_units] ])
      translated = cv2.warpAffine(images[i], M, (_w, _h), borderValue=(255,255,255))
      images_new[i] = translated
      translations.append((x_units, y_units))
    return images_new, translations

def random_translate_raw(images, units):
    units_x, units_y = units
    images_new = []
    translations = []
    
    for i in range(len(images)):
      _h, _w = tuple(images[i].shape)
      units_x = random.randint(0, units_x)
      units_y = random.randint(0, units_y)
      flip_x = random.randint(-1, 1)
      flip_y = random.randint(-1, 1)
      x_units = int(flip_x * units_x)
      y_units = int(flip_y * units_y)
      # units_x = random.uniform(-1.0*units_x, units_x)
      # units_y = random.uniform(-1.0*units_y, units_y)
      M = np.float32([ [1,0,x_units], [0,1,y_units] ])
      translated = cv2.warpAffine(images[i], M, (_w, _h), borderValue=(255,255,255))
      images_new.append(translated)
      translations.append((x_units, y_units))
    return np.array(images_new), translations


'''def random_brightness(images, max_brightness):
    # n, _h, _w = images.shape[:3]
    n = images.shape[0]
    images_new = images.copy()
    brightness = max_brightness * np.random.rand(n)
    for i in range(n):
        mask = (255 - images[i]) < brightness[i]
        images_new[i] = np.where((255 - images[i]) < brightness[i],255,images[i]+brightness[i])
    return images_new'''

def random_brightness(images, max_delta):
    # randomly adjust the brightness of the image by -max delta to max_delta
    images_new = images
    for i in range(images_new.shape[0]):
        delta = random.uniform(-1.0*max_delta, max_delta)
        images_new[i] = np.clip(images[i] + delta, 0, 255)
    return images_new

def random_rotate(images, degrees, u=0.4):
    # randomly rotate from 0 to degrees and from 0 to negative degrees
    images_new = images
    ret_degs = []
    
    for i in range(images_new.shape[0]):
      if np.random.random() < u:
          _h, _w = tuple(images[i].shape)
          deg = np.clip(random.uniform(-1.0*degrees, degrees), -1.0*degrees, degrees)
          ret_degs.append(deg)
          if deg < 0:
            deg = 360 + deg
          image_center = (_w/2, _h/2)
          rotation_mat = cv2.getRotationMatrix2D((_w/2, _h/2), int(deg), 1.0)
          abs_cos = abs(rotation_mat[0,0])
          abs_sin = abs(rotation_mat[0,1])
          # find the new width and height bounds
          bound_w = int(_h * abs_sin + _w * abs_cos)
          bound_h = int(_h * abs_cos + _w * abs_sin)
          # subtract old image center (bringing image back to origo) and adding the new image center coordinates
          rotation_mat[0, 2] += bound_w/2 - image_center[0]
          rotation_mat[1, 2] += bound_h/2 - image_center[1]
          # rotate image with the new bounds and translated rotation matrix
          # images_new[i] = cv2.warpAffine(images[i], rotation_mat, (bound_w, bound_h), borderValue=(255,255,255))
          M = cv2.getRotationMatrix2D((_w/2, _h/2), int(deg), 1)
          images_new[i] = cv2.warpAffine(images[i], M, (_w, _h), borderValue=(255,255,255))
        # images_new = center_crop_raw(images_new, (640, 640))
    return images_new

def random_rotate_raw(images, degrees):
    # randomly rotate from 0 to degrees and from 0 to negative degrees
    images_new = []
    ret_degs = []
    
    for i in range(len(images)):
      _h, _w = tuple(images[i].shape)
      deg = np.clip(random.uniform(-1.0*degrees, degrees), -1.0*degrees, degrees)
      # deg = np.random.choice([0, -30, -60, -90, -120, -180, 30, 60, 90, 120, 180])
      ret_degs.append(deg)
      if deg < 0:
        deg = 360 + deg

      image_center = (_w/2, _h/2)
      rotation_mat = cv2.getRotationMatrix2D((_w/2, _h/2), int(deg), 1.0)
      abs_cos = abs(rotation_mat[0,0])
      abs_sin = abs(rotation_mat[0,1])
      # find the new width and height bounds
      bound_w = int(_h * abs_sin + _w * abs_cos)
      bound_h = int(_h * abs_cos + _w * abs_sin)
      # subtract old image center (bringing image back to origo) and adding the new image center coordinates
      rotation_mat[0, 2] += bound_w/2 - image_center[0]
      rotation_mat[1, 2] += bound_h/2 - image_center[1]
      # rotate image with the new bounds and translated rotation matrix
      img = cv2.warpAffine(images[i], rotation_mat, (bound_w, bound_h), borderValue=(255,255,255))
      images_new.append(img)
    return np.array(images_new), ret_degs

def random_shear(images, intensity_range=(-0.5, 0.5), u=0.5):
    for i, img in enumerate(images):
        if np.random.random() < u:
            sh = np.random.uniform(-intensity_range[0], intensity_range[1])
            images[i] = Keras_Image_Proc.apply_affine_transform(np.stack((img,)*3, axis=-1), shear=sh)[:,:,0]
    return images



def random_shift(images, max_ratio):
    n, _h, _w = images.shape[:3]
    pad_x = int(_w * max_ratio) + 1
    pad_y = int(_h * max_ratio) + 1
    images_temp = padding(images, (pad_y, pad_x))
    images_new = images.copy()

    shift_x = (_w * max_ratio * np.random.rand(n)).astype(np.int32)
    shift_y = (_h * max_ratio * np.random.rand(n)).astype(np.int32)

    for i in range(n):
        images_new[i] = images_temp[i, pad_y+shift_y[i]:pad_y+shift_y[i]+_h, 
                            pad_x+shift_x[i]:pad_x+shift_x[i]+_w]

    return images_new    
    
'''def random_rotate(images, max_degree):
    images = images.astype(np.float32)

    n, _h, _w = images.shape[:3]
    images_new = images.copy()
    
    degree = max_degree * np.random.rand(n)

    for i in range(n):
        M = cv2.getRotationMatrix2D((_w/2, _h/2), int(degree[i]), 1)
        images_new[i] = cv2.warpAffine(images[i], M, (_w, _h), borderValue=(255,255,255))

    return images_new

def random_rotate_raw(images, max_degree):
    # images = images.astype(np.float32)

    n = images.shape[0]
    images_new = images.copy()
    
    degree = max_degree * np.random.rand(n)

    for i in range(n):
        _w, _h = images[i].shape[0], images[i].shape[1] 
        M = cv2.getRotationMatrix2D((_w/2, _h/2), int(degree[i]), 1)
        images_new[i] = cv2.warpAffine(images[i], M, (_w, _h))

    return images_new'''


def random_blur(images, blur_type, max_size):
    n, _h, _w = images.shape[:3]
    images_new = images.copy()
    
    kernel_size = max_size * np.random.rand(n)
    
    for i in range(n):
        size = int(kernel_size[i])
        if size > 0:
            if blur_type == 'motion':
                kernel = np.zeros((size, size))
                kernel[int((size-1)/2), :] = np.ones(size)
                kernel = kernel / size
                img = cv2.filter2D(images[i], -1, kernel)
            elif blur_type == 'gaussian':
                size = size // 2 * 2 + 1
                img = cv2.GaussianBlur(images[i], (size,size), 0)
            else:
                raise ValueError('Unkown blur type: {}'.format(blur_type))
            images_new[i] = img

    return images_new
    
def random_noise(images, stddev, min_=-1.0, max_=1.0, u=0.3):
    noises = np.random.normal(0.0, stddev, images.shape)
    for i, im in enumerate(images):
        if np.random.random() < u:
            images[i] = np.maximum(min_, np.minimum(max_, images[i] + noises[i]))
    return images

def random_downsample(images, min_ratio):
    n, _h, _w = images.shape[:3]
    images_new = images.copy()
    ratios = min_ratio + (1-min_ratio) * np.random.rand(n)

    for i in range(n):
        w = int(round(ratios[i] * _w))
        h = int(round(ratios[i] * _h))
        images_new[i,:h,:w] = misc.imresize(images[i], (h,w))
        images_new[i] = misc.imresize(images_new[i,:h,:w], (_h,_w))
        
    return images_new

def random_morph(images, u=0.5):
    for i, img in enumerate(images):
        if np.random.random() < u:
            area = img.shape[0] * img.shape[1]

            sl = 0.02
            sh = 0.2
            r1 = 0.3
           
            target_area = random.uniform(sl, sh) * area
            aspect_ratio = random.uniform(r1, 1/r1)

            
            kernel = np.ones((3,3), np.uint8)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.shape[1] and h < img.shape[0]:
                x1 = random.randint(0, img.shape[0] - h)
                y1 = random.randint(0, img.shape[1] - w)

                img[x1:x1+h, y1:y1+w] = cv2.erode(cv2.dilate(img[x1:x1+h, y1:y1+w], kernel, iterations=1),
                    kernel, iterations=1)
                images[i] = img
    return images

def random_interpolate(images):
    _n, _h, _w = images.shape[:3]
    nd = images.ndim - 1
    assert _n % 2 == 0
    n = int(_n / 2)

    ratios = np.random.rand(n,*([1]*nd))
    images_left, images_right = (images[np.arange(n)*2], images[np.arange(n)*2+1])
    images_new = ratios * images_left + (1-ratios) * images_right
    images_new = images_new.astype(np.uint8)

    return images_new
    
def expand_flip(images):
    '''Flip each image in the array and insert it after the original image.'''
    _n, _h, _w = images.shape[:3]
    shape_new = get_new_shape(images, n=2*_n)
    images_new = np.stack([images, flip(images)], axis=1)
    images_new = images_new.reshape(shape_new)
    return images_new

def five_crop(images, size):
    _n, _h, _w = images.shape[:3]
    h, w = tuple(size)
    assert h <= _h and w <= _w

    shape_new = get_new_shape(images, size, n=5*_n)
    images_new = []
    images_new.append(images[:,:h,:w])
    images_new.append(images[:,:h,-w:])
    images_new.append(images[:,-h:,:w])
    images_new.append(images[:,-h:,-w:])
    images_new.append(center_crop(images, size))
    images_new = np.stack(images_new, axis=1).reshape(shape_new)
    return images_new

def ten_crop(images, size):
    _n, _h, _w = images.shape[:3]
    shape_new = get_new_shape(images, size, n=10*_n)
    images_ = five_crop(images, size)
    images_flip_ = five_crop(flip(images), size)
    images_new = np.stack([images_, images_flip_], axis=1)
    images_new = images_new.reshape(shape_new)
    return images_new


def FastEnhanceTexture(img,sigma=2.5):
    img = img.astype(np.float32)
    h, w = img.shape
    h2 = 2 ** nextpow2(h)
    w2 = 2 ** nextpow2(w)

    FFTsize = np.max([h2, w2])
    x, y = np.meshgrid(range(-FFTsize / 2, FFTsize / 2), range(-FFTsize / 2, FFTsize / 2))
    r = np.sqrt(x * x + y * y) + 0.0001
    r = r/FFTsize

    L = 1. / (1 + (2 * math.pi * r * sigma)** 4)
    img_low = LowpassFiltering(img, L)

    gradim1=  compute_gradient_norm(img)
    gradim1 = LowpassFiltering(gradim1,L)

    gradim2=  compute_gradient_norm(img_low)
    gradim2 = LowpassFiltering(gradim2,L)

    diff = gradim1-gradim2
    ar1 = np.abs(gradim1)
    diff[ar1>1] = diff[ar1>1]/ar1[ar1>1]
    diff[ar1 <= 1] = 0

    cmin = 0.3
    cmax = 0.7

    weight = (diff-cmin)/(cmax-cmin)
    weight[diff<cmin] = 0
    weight[diff>cmax] = 1


    u = weight * img_low + (1-weight)* img

    temp = img - u

    lim = 20

    temp1 = (temp + lim) * 255 / (2 * lim)

    temp1[temp1 < 0] = 0
    temp1[temp1 >255] = 255
    v = temp1
    return v

def compute_gradient_norm(input):
    input = input.astype(np.float32)

    Gx, Gy = np.gradient(input)
    out = np.sqrt(Gx * Gx + Gy * Gy) + 0.000001
    return out

def LowpassFiltering(img,L):
    h,w = img.shape
    h2,w2 = L.shape

    img = cv2.copyMakeBorder(img, 0, h2-h, 0, w2-w, cv2.BORDER_CONSTANT, value=0)

    img_fft = np.fft.fft2(img)
    img_fft = np.fft.fftshift(img_fft)

    img_fft = img_fft * L
    rec_img = np.fft.ifft2(np.fft.fftshift(img_fft))
    rec_img = np.real(rec_img)
    rec_img = rec_img[:h,:w]

    return rec_img

def nextpow2(x):
    return int(math.ceil(math.log(x, 2)))


register = {
    # 'fast_enhance_texture': fast_enhance_texture,
    'resize': resize,
    'padding': padding,
    'flip': flip,
    'random_crop': random_crop,
    'center_crop': center_crop,
    'random_flip': random_flip,
    'standardize': standardize_images,
    'random_shift': random_shift,
    'random_interpolate': random_interpolate,
    'random_rotate': random_rotate,
    'random_rotate_raw': random_rotate_raw,
    'random_scale': random_scale,
    'random_blur': random_blur,
    'random_noise': random_noise,
    'random_downsample': random_downsample,
    'random_morph': random_morph,
    'expand_flip': expand_flip,
    'five_crop': five_crop,
    'ten_crop': ten_crop,
    'segment_fp': segment_fp,
    'scale_fp': scale_fp,
    'random_shear': random_shear,
    'random_contrast': random_contrast,
    'center_crop_raw': center_crop_raw,
    'top_crop_raw': top_crop_raw,
    'random_translate': random_translate,
    'random_translate_raw': random_translate_raw,
    'resize_raw': resize_raw,
    'random_brightness': random_brightness,
    'top_center_crop': top_center_crop,
    'contrast_enhance': contrast_enhance
}

def preprocess(images, config, is_training=False, load_only=False):
    ''' Legacy function. Equaivalent to batch_process but it uses config module as input '''
    # Load images first if they are file paths
    if type(images[0]) == str:
        image_paths = images
        images = []
        assert (config.channels==1 or config.channels==3)
        mode = 'RGB' if config.channels==3 else 'LA'
        for image_path in image_paths:
            try:
                images.append(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY))
            except:
                print('FAILEEEEEED: {}'.format(image_path))
                continue
        # images = np.stack(images, axis=0)
        images = np.array(images)
    if load_only:
        return images
    # Process images
    proc_funcs = config.preprocess_train if is_training else config.preprocess_test
    props = {}
    for proc in proc_funcs:
        proc_name, proc_args = proc[0], proc[1:]
        assert proc_name in register, \
            "Not a registered preprocessing function: {}".format(proc_name)
        images = register[proc_name](images, *proc_args)
        if len(images) == 2:
            props[proc_name] = images[1]
            images = images[0]
    if len(images.shape) == 3:
        images = images[:,:,:,None]
    if len(list(props.keys())) > 0:
        return images, props
        # return images
    else:
        return images
        

def batch_process(images, proc_funcs, channels=3):
    # Load images first if they are file paths
    if type(images[0]) == str:
        image_paths = images
        images = []
        assert (channels==1 or channels==3)
        mode = 'RGB' if channels==3 else 'I'
        for image_path in image_paths:
            images.append(misc.imread(image_path, mode=mode))
        images = np.stack(images, axis=0)
    else:
        assert type(images[0]) is np.array, \
            'Illegal input type for images: {}'.format(type(images[0]))

    # Process images
    for proc in proc_funcs:
        proc_name, proc_args = proc[0], proc[1:]
        assert proc_name in register, \
            "Not a registered preprocessing function: {}".format(proc_name)
        images = register[proc_name](images, *proc_args)
    if len(images.shape) == 3:
        images = images[:,:,:,None]
    return images
        
