import pandas as pd
import cv2
import numpy as np
from PIL import Image, ImageCms
from skimage import color, io


def patchify_WSI(img_path, case_id):


    sliding_window = 3600
    patch = 256
    image = cv2.imread(img_path + str(case_id))
    # image = Image.open(img_path + str(case_id))
    # imarray = numpy.array(image)
    # imarray.shape
    # image.show()

    height = image.shape[0]
    width = image.shape[1]

    num_vertical_tiles = height / patch
    num_horizontal_tiles = width / patch


    lab_histograms = []
    lbp_histograms = []

    for v in range(num_vertical_tiles):
        for h in range(num_horizontal_tiles):

            cropped_image = image[80:280, 150:330]
            cv2.imwrite("Cropped Image.jpg", cropped_image)
            # image_lab = applycform( image, makecform('srgb2lab'));
            # lbp = imread([lbp_path num2str(case_id) '_ns_lbp.tiff'], 'PixelRegion', region);




def color_features():

    # Open image and discard alpha channel which makes wheel round rather than square
    im = Image.open('images/test.png').convert('RGB')

    # Convert to Lab colourspace
    srgb_p = ImageCms.createProfile("sRGB")
    lab_p  = ImageCms.createProfile("LAB")

    rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
    Lab = ImageCms.applyTransform(im, rgb2lab)

    L, a, b = Lab.split()
    print(type(L))

    L.save('images/L.png')
    a.save('images/a.png')
    b.save('images/b.png')

    IL = np.asarray(Image.open('images/L.png'))

    IL1 = Image.fromarray(np.uint8(IL))


    print(type(IL))
    print((IL.shape))

    # Open image and make Numpy arrays 'rgb' and 'Lab'
    # rgb = io.imread('images/test.png')
    Lab = color.rgb2lab(im)
    L1 = Lab[:,:,0]

    a1 = Lab[:,:,1]

    b1 = Lab[:,:,2]
    print(type(L1))
    print((L1.shape))

    # L1.save('images/L1.png')
    # a1.save('images/a1.png')
    # b1.save('images/b1.png')

    print(IL1 == L1)
    print(a == a1)
    print(b == b1)

    return 1

color_features()

def texture_features():

    return 1


