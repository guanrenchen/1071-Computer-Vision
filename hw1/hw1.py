# -*- coding: utf-8 -*-

import sys
from hw1_ui import Ui_MainWindow
import cv2
from PyQt5.QtWidgets import QMainWindow, QApplication
import numpy as np

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.onBindingUI()

    # Write your code below
    # UI components are defined in hw1_ui.py, please take a look.
    # You can also open hw1.ui by qt-designer to check ui components.
    
    def onBindingUI(self):
        self.btn1_1.clicked.connect(self.on_btn1_1_click)
        self.btn1_2.clicked.connect(self.on_btn1_2_click)
        self.btn1_3.clicked.connect(self.on_btn1_3_click)
        self.btn1_4.clicked.connect(self.on_btn1_4_click)
        self.btn2_1.clicked.connect(self.on_btn2_1_click)
        self.btn3_1.clicked.connect(self.on_btn3_1_click)
        self.btn4_1.clicked.connect(self.on_btn4_1_click)
        self.btn4_2.clicked.connect(self.on_btn4_2_click)
        self.btn5_1.clicked.connect(self.on_btn5_1_click)
        self.btn5_2.clicked.connect(self.on_btn5_2_click)
    
    # button for problem 1.1
    def on_btn1_1_click(self):
        img = cv2.imread('images/dog.bmp')
        cv2.imshow('Dog', img)
        h, w = img.shape[:2]
        print('Height = {h}\nWidth = {w}'.format(h=h, w=w))

    def on_btn1_2_click(self):
        img = cv2.imread('images/color.png')
        cv2.imshow('Source', img)
        cv2.imshow('BGR to RBG', img[:,:,[1,2,0]])

    def on_btn1_3_click(self):
        img = cv2.imread('images/dog.bmp')
        cv2.imshow('Source', img)
        cv2.imshow('Horizontal flip', cv2.flip(img, 1))

    def on_btn1_4_click(self):
        img1 = cv2.imread('images/dog.bmp')
        img2 = cv2.flip(img1, 1)

        def on_trackbar_blend_changed(pos):
            percent = pos / 100
            img = cv2.addWeighted(img1, percent, img2, 1-percent, 0)
            cv2.imshow('BLEND', img)
            return pos
        on_trackbar_blend_changed(0)
        cv2.createTrackbar('Blend : ', 'BLEND', 0, 100, on_trackbar_blend_changed)
        
    def on_btn2_1_click(self):
        img = cv2.imread('images/M8.jpg', cv2.IMREAD_GRAYSCALE)
        img = cv2.GaussianBlur(img,(3,3),0)

        sobelX = np.array([ [-1,0,1], [-2,0,2], [-1,0,1] ])
        sobelY = np.array([ [-1,-2,-1], [0,0,0], [1,2,1] ])
        v_edge = np.zeros(img.shape)
        h_edge = np.zeros(img.shape)
        angles = np.zeros(img.shape)

        for i in range(img.shape[0] - 2):
            for j in range(img.shape[1] - 2):
                x = i+1
                y = j+1
                sub_matrix = img[i:i+3, j:j+3]
                gx = np.sum(sobelX * sub_matrix)
                gy = np.sum(sobelY * sub_matrix)
                angles[x][y] = np.arctan2(gy, gx)
                v_edge[x][y] = abs(gx)
                h_edge[x][y] = abs(gy)

        threshold = 40 / 255

        mag_img = np.sqrt(v_edge**2 + h_edge**2)
        mag_img /= np.amax(mag_img)

        v_edge /= np.amax(v_edge)
        v_edge *= v_edge > threshold
        h_edge /= np.amax(h_edge)
        h_edge *= h_edge > threshold

        cv2.imshow('Source', img)
        cv2.imshow('vertical edge', v_edge)
        cv2.imshow('horizontal edge', h_edge)

        def on_trackbar_magnitude_changed(threshold=40):
            print(threshold)
            cv2.imshow('Magnitude', mag_img * (mag_img > (threshold / 255)))
        on_trackbar_magnitude_changed()
        cv2.createTrackbar('Threshold : ', 'Magnitude', 0, 255, on_trackbar_magnitude_changed)

        dir_img = mag_img * (mag_img > threshold)
        def on_trackbar_direction_changed(theta=10, offset=10):
            print(theta)
            lower = ((theta - offset) % 360 - 180) * np.pi / 180
            upper = ((theta + offset) % 360 - 180) * np.pi / 180
            func = np.bitwise_and if lower < upper else np.bitwise_or
            bools = func(angles > lower, angles < upper) 
            cv2.imshow('Direction', dir_img * bools)
        on_trackbar_direction_changed()
        cv2.createTrackbar('Angle : ', 'Direction', 0, 360, on_trackbar_direction_changed)

    def on_btn3_1_click(self):
        from cv2 import pyrDown, pyrUp, subtract, add
        g_lv0 = cv2.imread('images/pyramids_Gray.jpg', 0)
        g_lv1 = pyrDown(g_lv0)
        g_lv2 = pyrDown(g_lv1)

        l_lv0 = subtract(g_lv0, pyrUp(g_lv1))
        l_lv1 = subtract(g_lv1, pyrUp(g_lv2))

        i_lv2 = g_lv2
        i_lv1 = add(l_lv1, pyrUp(i_lv2))
        i_lv0 = add(l_lv0, pyrUp(i_lv1))

        cv2.imshow('Gaussian level 1', g_lv1)
        cv2.imshow('Laplacian level 0', l_lv0 / np.amax(l_lv0))
        cv2.imshow('Inverse level 1', i_lv1)
        cv2.imshow('Inverse level 0', i_lv0)

    def on_btn4_1_click(self):
        img = cv2.imread('images/QR.png', cv2.IMREAD_GRAYSCALE)
        threshold, dst = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
        cv2.imshow('Original image', img)
        cv2.imshow('Global threshold image', dst)

    def on_btn4_2_click(self):
        img = cv2.imread('images/QR.png', cv2.IMREAD_GRAYSCALE)
        dst = cv2.adaptiveThreshold(
            src = img, 
            maxValue = 255, 
            adaptiveMethod = cv2.ADAPTIVE_THRESH_MEAN_C, 
            thresholdType = cv2.THRESH_BINARY,
            blockSize = 19,
            C = -1,
            )
        cv2.imshow('Original image', img)
        cv2.imshow('Adaptive threshold image', dst)

    def on_btn5_1_click(self):
        # edtAngle, edtScale. edtTx, edtTy to access to the ui object
        def getFloatValue(x):
            try:
                return float(x.text())
            except Exception as ex:
                print(ex)
                return 0
        deg = getFloatValue(self.edtAngle)
        scale = getFloatValue(self.edtScale)
        tx = getFloatValue(self.edtTx)
        ty = getFloatValue(self.edtTy)

        img = cv2.imread('images/OriginalTransform.png')
        M = cv2.getRotationMatrix2D((130,125), deg, scale)
        print(M)
        M[0][2] += tx
        M[1][2] += ty
        print(M)
        h, w = img.shape[:2]

        cv2.imshow('transform', cv2.warpAffine(img, M, (w, h)))

    def on_btn5_2_click(self):
        img = cv2.imread('images/OriginalPerspective.png')
        ref_point = []
        def setRefPoint(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONUP:
                ref_point.append([x, y])
                print([x,y])
                if len(ref_point) < 4:
                    return
                dst_point = np.float32([[20,20],[450,20],[450,450],[20,450]])
                M = cv2.getPerspectiveTransform(np.float32(ref_point), dst_point)
                dst = cv2.warpPerspective(img, M, (450,450))
                cv2.imshow('perspective', dst)
                ref_point.clear()

        cv2.imshow('original', img)
        cv2.setMouseCallback('original', setRefPoint)
    ### ### ###


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
