# -*- coding: utf-8 -*-
"""
pre-processing and pattern matching.
This python module can perform the following functions:
1. Binarization - method binary_img(img) performs this function
2. Skew correction - method skew_correction(img) performs this function
Need to introduce machine learning of some sort to make the skew correction
method run faster :(
Or... A simple fix would be to resize the image first, and then apply the skew
correction method! That'll probably take lesser time...
Resizing is yielding better results.
"""

import logging

import cv2
import numpy as np
from scipy.stats import mode

logging.basicConfig(
  level=logging.DEBUG,
  format="%(levelname)s: %(asctime)s {%(filename)s:%(lineno)d}: %(message)s "
)

kernel = np.ones((5, 5), np.uint8)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

"""
Method to binarize an image
Input: Grayscale image
Output: Binary image
The nature of the output is such that the text(foreground) has a colour 
value of (255,255,255), and the background has a value of (0,0,0).
"""


def binary_img(img):
  # img_erode = cv2.dilate(img,kernel,iterations = 2)
  blur = cv2.medianBlur(img, 5)

  # mask1 = np.ones(img.shape[:2],np.uint8)
  """Applying histogram equalization"""
  cl1 = clahe.apply(blur)

  circles_mask = cv2.dilate(cl1, kernel, iterations=1)
  circles_mask = (255 - circles_mask)

  thresh = 1
  circles_mask = cv2.threshold(circles_mask, thresh, 255, cv2.THRESH_BINARY)[1]

  edges = cv2.Canny(cl1, 100, 200)

  edges = cv2.bitwise_and(edges, edges, mask=circles_mask)

  dilation = cv2.dilate(edges, kernel, iterations=1)

  display = cv2.bitwise_and(img, img, mask=dilation)

  cl2 = clahe.apply(display)
  cl2 = clahe.apply(cl2)

  ret, th = cv2.threshold(cl2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  th = 255 - th

  thg = cv2.adaptiveThreshold(display, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                              cv2.THRESH_BINARY, 11, 2)

  # final = cv2.bitwise_and(dilation,dilation,mask=th)

  finalg = cv2.bitwise_and(dilation, dilation, mask=thg)

  finalg = 255 - finalg

  abso = cv2.bitwise_and(dilation, dilation, mask=finalg)

  return abso


"""
Method to resize the image. This is going to help in reducing the number 
of computations, as the size of data will reduce.
"""


def resize(img):
  r = 1000.0 / img.shape[1]
  dim = (1000, int(img.shape[0] * r))
  resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

  # cv2.imshow('resized', resized)
  return resized


"""
Method to correct the skew of an image
Input: Binary image
Output: Skew corrected binary image
The nature of the output is such that the binary image is rotated appropriately
to remove any angular skew.
Find out the right place to insert the resizing method call.
Try to find one bounding rectangle around all the contours
"""


def skew_correction(img):
  areas = []  # stores all the areas of corresponding contours
  dev_areas = []  # stores all the areas of the contours within 1st std deviation in terms of area#stores all the white pixels of the largest contour within 1st std deviation
  all_angles = []
  k = 0

  binary = binary_img(img)
  # binary = resize(binary)
  im2, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  # cnt = contours[0]
  # upper_bound=len(contours)
  height_orig, width_orig = img.shape[:2]
  words = np.zeros(img.shape[:2], np.uint8)

  for c in contours:
    areas.append(cv2.contourArea(c))

  std_dev = np.std(areas)
  for i in areas:
    dev_areas.append(i - std_dev)

  dev_contours = np.zeros(img.shape[:2], np.uint8)

  for i in dev_areas:
    if ((i > (-std_dev)) and (i <= (std_dev))):
      cv2.drawContours(dev_contours, contours, k, (255, 255, 255), -1)
    k += 1

  sobely = cv2.Sobel(dev_contours, cv2.CV_64F, 0, 1, ksize=5)
  abs_sobel64f = np.absolute(sobely)
  sobel_8u = np.uint8(abs_sobel64f)

  # cv2.imshow('Output2',sobel_8u)

  minLineLength = 100
  maxLineGap = 10
  lines = cv2.HoughLinesP(sobel_8u, 1, np.pi / 180, 100, minLineLength, maxLineGap)

  for x1, y1, x2, y2 in lines[0]:
    cv2.line(words, (x1, y1), (x2, y2), (255, 255, 255), 2)
  # cv2.imshow('hough',words)

  height_orig, width_orig = img.shape[:2]
  all_angles = []

  im2, contours, hierarchy = cv2.findContours(words, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  logging.debug(len(contours))
  contour_count = 0
  for c in contours:
    # max_index = np.argmax(areas)
    # current_contour = np.zeros(img.shape[:2],np.uint8)
    current_contour = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(current_contour, contours, contour_count, (255, 255, 255), -1)

    height, width = current_contour.shape[:2]

    # all_white_pixels = []
    current_white_pixels = []

    for i in range(0, height):
      for j in range(0, width):
        if (current_contour.item(i, j) == 255):
          current_white_pixels.append([i, j])

    matrix = np.array(current_white_pixels)

    """Finding covariance matrix"""
    C = np.cov(matrix.T)

    eigenvalues, eigenvectors = np.linalg.eig(C)

    """Finding max eigenvalue"""
    # max_ev = max(eigenvalues)
    """Finding index of max eigenvalue"""
    max_index = eigenvalues.argmax(axis=0)

    """The largest eigen value gives the approximate length of the bounding
        ellipse around the largest word. If we follow the index of the largest 
        eigen value and find the eigen vectors in the column of that index,
        we'll get the x and y coordinates of it's centre."""
    y = eigenvectors[1, max_index]
    x = eigenvectors[0, max_index]

    angle = (np.arctan2(y, x)) * (180 / np.pi)
    all_angles.append(angle)
    contour_count += 1
    logging.debug(contour_count)

    logging.debug(all_angles)
    angle = np.mean(all_angles)
    logging.debug(angle)

  k = 0
  non_zero_angles = []

  for i in all_angles:
    if ((i != 0) and (i != 90.0)):
      non_zero_angles.append(i)

  logging.debug(non_zero_angles)

  rounded_angles = []
  for i in non_zero_angles:
    rounded_angles.append(np.round(i, 0))

  logging.debug(rounded_angles)
  logging.debug("mode is")
  # logging.debug(np.mode(rounded_angles))
  # angle = np.mean(non_zero_angles)
  # angle = np.mode(rounded_angles)

  mode_angle = mode(rounded_angles)[0][0]
  logging.debug(mode_angle)

  precision_angles = []
  for i in non_zero_angles:
    if (np.round(i, 0) == mode_angle):
      precision_angles.append(i)

  logging.debug('precision angles:')
  logging.debug(precision_angles)

  angle = np.mean(precision_angles)
  logging.debug('Finally, the required angle is:')
  logging.debug(angle)

  # M = cv2.getRotationMatrix2D((width/2,height/2),-(90+angle),1)
  M = cv2.getRotationMatrix2D((width / 2, height / 2), -(90 + angle), 1)
  dst = cv2.warpAffine(img, M, (width_orig, height_orig))

  # cv2.imshow('final',dst)
  cv2.imwrite('skewcorrected2.jpg', dst)

  return dst


def preprocess(img):
  return skew_correction(img)

# Does not work with linux:
# cv2.destroyAllWindows()
