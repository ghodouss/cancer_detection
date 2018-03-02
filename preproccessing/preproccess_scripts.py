import cv2
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes




def gray_and_blur(img, k_size=13):
    """
    Module to convert image to grayscale and
    to blur for easier edge detection

    Inputs
    ---------------
    img: A RGB numpy file

    Outputs
    --------------
    img: 3-D Grayscale numpy file
    """  
    #convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #blur image
    img = cv2.GaussianBlur(img, (k_size, k_size), 0)
    
    return img

def pre_edge(imgs, k_size=13):

    """
    runs gray and blur for np images
    in a list and returns new list

    Inputs
    ----------------
    img: list of numpy RGB img files

    Outputs
    -----------------------
    proccessed: list of gray and blurred 3-D np files
    """
    processed = []
    
    for img in imgs:
        
        processed.append(gray_and_blur(img, k_size))
        
    return processed


def get_edges(imgs, low_t=40, high_t=120, k_size=17):
    """
    given a list of images, run through the
    images and returns a list of images
    run through Canny Edge Detection

    Inputs
    -----------------
    imgs: list of np RGB img files

    Outputs
    ---------------
    edged: list of 3-D np Canny-edge img files 
    """

    edged = []
    #process images
    processed = pre_edge(imgs, k_size=k_size)
    
    for img in processed:
        edged.append(cv2.Canny(img, low_t, high_t))
        
    return edged


def get_cancer_mask(img):
    """
    Pass in a single image and returns
    a mask image with 1's at the location of the
    cancer mole and 0's everywhere else.

    Inputs
    ------------------
    img : RGB NP image file

    Outputs
    ------------------
    mask : a 3-D NP file that provides 0-1 mask values
    """
    
    # Copies image to not effect original img.
    img = img.copy()
    
    # Gets canny edges for the img.
    edge = get_edges([img])[0]
    
    # Get the contours from the edged image.
    _ , contours, hierarchy = cv2.findContours(edge,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the areas and choose the contour that has the max area:
    areas = []
    
    # getting areas,
    for cntr in contours:
        areas.append(cv2.contourArea(cntr))
    
    # and getting biggest contour.
    max_index = (areas.index(max(areas)))
    max_cntr = contours[max_index]
    
    # Load blank imag and draw the max contour on the image in white.
    img = np.ones(img.shape)
    img = cv2.drawContours(img, max_cntr, -1, (255,255,255), 3)
    
    # Run blur to close the contour.
    se = np.ones((9,9), dtype='uint8')
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se)
    
    # Set min values to zero, and convert to bool.
    img = ((img-img.min())/255).astype(bool)
    
    # Use the bool values to create a mask
    mask = binary_fill_holes(img[:,:,0].astype(bool)).reshape(img.shape[0], img.shape[1], 1)

    
    return mask.astype(np.int8)



def get_cancer_mole(img):
    """
    Function that gets only the cancer mole from an image
    and returns the rest of the image pixels set to 0

    Inputs
    --------------
    img : RGB numpy image

    Outputs
    ----------------
    img : RGB numpy image after masking
    """
    
    mask = get_cancer_mask(img)
    
    res = cv2.bitwise_and(img, img, mask=mask)
    
    return res


def juxtapose_mole_and_background(cancer_img, background_img):
    
    """
    Given an unprocessed cancer image and background image,
    gets the cancer mole and superimposeses it on the background
    image, and returns the juxtaposed image

    Inputs
    ----------------------------------
    cancer_image : RGB 3-D NP array
    background_image : RGB 3-D NP array

    Output
    -------------------
    Juxtaposed_image: Background with Mole superimpsed; RGB 3-D NP Array
     """


    mask = get_cancer_mask(cancer_img)
    
    mole = cv2.bitwise_and(cancer_img, cancer_img, mask=mask)

    mask = 1 - mask
    
    cleared_background = cv2.bitwise_and(background_img, background_img, mask=mask)
    
    mixed_img = cv2.bitwise_or(mole, cleared_background)
    
    return mixed_img