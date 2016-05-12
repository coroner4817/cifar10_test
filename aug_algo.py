import cv2
import numpy as np
import random as rd


class aug_algo:

    def __init__(self):
        pass

    def horizShiftLeft(self, image, param1, param2):
        rows, cols, channels = image.shape
        shift = rd.randint(int(cols*param1), int(cols*param2))
        ROI = image[0:rows, (0+shift):cols]
        endCol = image[:, cols-1, :]
        endCol = endCol[:, np.newaxis, :]
        added = np.tile(endCol, (1, shift, 1))
        ROI = np.concatenate((ROI, added), axis=1)
        return ROI


    def horizShiftRight(self, image, param1, param2):
        rows, cols, channels = image.shape
        shift = rd.randint(int(cols*param1), int(cols*param2))
        ROI = image[0:rows, 0:(cols-shift)]
        StartCol = image[:, 0, :]
        StartCol = StartCol[:, np.newaxis, :]
        added = np.tile(StartCol, (1, shift, 1))
        ROI = np.concatenate((added, ROI), axis=1)
        return ROI


    def vertiShiftDown(self, image, param1, param2):
        rows, cols, channels = image.shape
        shift = rd.randint(int(rows*param1), int(rows*param2))
        ROI = image[0:(rows-shift), 0:cols]
        startRow = image[0, :, :]
        startRow = startRow[np.newaxis, :, :]
        added = np.tile(startRow, (shift, 1, 1))
        ROI = np.concatenate((added, ROI), axis=0)
        return ROI

    def vertiShiftUp(self, image, param1, param2):
        rows, cols, channels = image.shape
        shift = rd.randint(int(rows*param1), int(rows*param2))
        ROI = image[(0+shift):rows, 0:cols]
        startRow = image[rows-1, :, :]
        startRow = startRow[np.newaxis, :, :]
        added = np.tile(startRow, (shift, 1, 1))
        ROI = np.concatenate((ROI, added), axis=0)
        return ROI


    def rotatedCW(self, image, param1, param2, param3):
        rows, cols, channels = image.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), rd.randint(int(-param2*90), int(-param1*90)), param3)
        rotated = cv2.warpAffine(image, M, (cols, rows))
        return rotated


    def rotatedCCW(self, image, param1, param2, param3):
        rows, cols, channels = image.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), rd.randint(int(param1*90), int(param2*90)), param3)
        rotated = cv2.warpAffine(image, M, (cols, rows))
        return rotated


    def cropSkretch(self, image, param1, param2):
        rows, cols, channels = image.shape
        x0 = rd.randint(int(param1*cols), int(param2*cols))
        y0 = rd.randint(int(param1*cols), int(param2*rows))
        ROI = image[(0+y0):(rows-y0), (0+x0):(cols-x0)]
        ROI = cv2.resize(ROI, (cols, rows))
        return ROI
