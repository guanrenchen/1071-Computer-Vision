# -*- coding: utf-8 -*-

import io, os, sys, glob

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QMainWindow, QApplication

from hw2_ui import Ui_MainWindow

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.onBindingUI()

        self.dir = 'images/CameraCalibration/'
        self.comboBox3_3.addItems([x[:-4] for x in os.listdir(self.dir)])

        w = 11
        h = 8
        self.corner_count = w*h
        self.board = (w,h)
        self.objp = np.zeros((w*h,3), np.float32)
        self.objp[:,:2] = np.flip(np.mgrid[0:w,0:h].T.reshape(-1,2), 0)

        imgpoints, objpoints = [], []

        for fname in glob.glob(self.dir+'*.bmp'):
            img = cv2.imread(fname)
            ret, corners = cv2.findChessboardCorners(img, self.board)
            if ret:
                objpoints.append(self.objp)
                imgpoints.append(corners)

        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = \
            cv2.calibrateCamera(objpoints, imgpoints, img.shape[:2], None, None)

    def onBindingUI(self):
        self.btn1_1.clicked.connect(self.on_btn1_1_click)
        self.btn1_2.clicked.connect(self.on_btn1_2_click)
        self.btn2_1.clicked.connect(self.on_btn2_1_click)
        self.btn2_2.clicked.connect(self.on_btn2_2_click)
        self.btn2_3.clicked.connect(self.on_btn2_3_click)
        self.btn3_1.clicked.connect(self.on_btn3_1_click)
        self.btn3_2.clicked.connect(self.on_btn3_2_click)
        self.btn3_3.clicked.connect(self.on_btn3_3_click)
        self.btn3_4.clicked.connect(self.on_btn3_4_click)
        self.btn4.clicked.connect(self.on_btn4_click)

    def on_btn1_1_click(self):
        img = cv2.imread('images/plant.jpg', cv2.IMREAD_GRAYSCALE)
        cv2.imshow('Original image', img)
        hist_img = my_hist(img, 'gray value', 'pixel number', [0,256])
        cv2.imshow('Original image histogram', hist_img)

    def on_btn1_2_click(self):
        img = cv2.imread('images/plant.jpg', cv2.IMREAD_GRAYSCALE)
        cv2.equalizeHist(img, img)
        cv2.imshow('Equalized image', img)
        hist_img = my_hist(img, 'gray value', 'pixel number', [0, 256])
        cv2.imshow('Equalized image histogram', hist_img)
            
    def on_btn2_1_click(self):
        img = cv2.imread('images/q2_train.jpg')
        cv2.imshow('Input image', img)

        circles = circle_detection(img)
        if circles is not None:
            for (x,y,r) in np.round(circles[0, :]).astype("int"):
                cv2.circle(img,(x,y),r,(0,255,0),1)
                cv2.circle(img,(x,y),2,(0,0,255),2)

        cv2.imshow('Detected circles',img)
        
    def on_btn2_2_click(self):
        img = cv2.imread('images/q2_train.jpg')
        cv2.imshow('Input image', img)

        mask = np.zeros(img.shape, dtype=np.uint8)
        circles = circle_detection(img)

        if circles is not None:
            for (x,y,r) in np.round(circles[0, :]).astype("int"):
                cv2.circle(mask,(x,y),r,(255,255,255),-1)
        gray_roi = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        hue = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,0][gray_roi>0]

        plt.clf()
        plt.xlabel('Angle')
        plt.ylabel('Probability')
        plt.axis([0,180,0,1])
        x,bins,p = plt.hist(hue, 180, color='red', width=1.0)
        x_max = max(x)
        for item in p:
            item.set_height(item.get_height() / x_max)
        
        with io.BytesIO() as buf:
            plt.savefig(buf, format='png')
            buf.seek(0)
            hist_img = cv2.imdecode(np.frombuffer(buf.getbuffer(), np.uint8), -1)
            cv2.imshow('Normalized Hue histogram', hist_img)
 
    def on_btn2_3_click(self):
        img = cv2.imread('images/q2_test.jpg')
        cv2.imshow('Input image', img)

        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        img_roi = np.zeros(img.shape, dtype=np.uint8)
        circles = circle_detection(img)
        if circles is not None:
            for (x,y,r) in np.round(circles[0,:]).astype("int"):
                cv2.circle(img_roi,(x,y),r,(255,255,255),-1)
                
        img_roi = img*(img_roi>0)
        hsv_roi = cv2.cvtColor(img_roi, cv2.COLOR_BGR2HSV)
        gray_roi = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)

        hist = cv2.calcHist([hsv_roi], [0, 1], gray_roi, [9, 71], [103, 121, 48, 190])
        backproj = cv2.calcBackProject([hsv_img], [0, 1], hist, [0, 180, 0, 256], 1)

        cv2.normalize(backproj, backproj, 0, 255, cv2.NORM_MINMAX)
        cv2.threshold(backproj, 100, 255, cv2.THRESH_BINARY, backproj)

        cv2.imshow('BackProjection_result', backproj)

    def on_btn3_1_click(self):
        imgs = {}
        for fname in os.listdir(self.dir):
            img = cv2.imread(self.dir + fname)
            ret, corners = cv2.findChessboardCorners(img, self.board)
            cv2.drawChessboardCorners(img, self.board, corners, ret)
            imgs[int(fname[:-4])] = img.copy()

        window_name = 'Chessboard Corners'
        def on_trackbar_corner_changed(pos):
            if pos > 0:
                cv2.imshow(window_name, imgs[pos])
        on_trackbar_corner_changed(1)
        cv2.createTrackbar('Index', window_name, 1, len(imgs), on_trackbar_corner_changed)

        while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
            cv2.waitKey(50)
        cv2.destroyAllWindows()

    def on_btn3_2_click(self):
        print(self.mtx)

    def on_btn3_3_click(self):
        img = cv2.imread(self.dir + self.comboBox3_3.currentText() + '.bmp')

        ret, corners = cv2.findChessboardCorners(img, self.board)
        flag, rvec, tvec = cv2.solvePnP(self.objp, corners, self.mtx, self.dist)

        ext = np.zeros((3,4), dtype=float)
        cv2.Rodrigues(np.array(rvec), ext[:,:3])
        ext[:,3:] = np.array(tvec)

        print(ext)

    def on_btn3_4_click(self):
        print(self.dist)

    def on_btn4_click(self):
        axis = np.float32([
            [0,0,0], [0,2,0], [2,2,0], [2,0,0],
            [0,0,-2],[0,2,-2],[2,2,-2],[2,0,-2]])
        color = (0,0,255)
        thickness = 5

        imgs = []

        for fname in glob.glob(self.dir+'[1-5].bmp'):
            img = cv2.imread(fname)

            ret, corners = cv2.findChessboardCorners(img, (11,8))
            ret, mtx, dist, rvecs, tvecs = \
                cv2.calibrateCamera([self.objp], [corners], img.shape[:2], None, None)

            imgpts, _ = cv2.projectPoints(axis, np.array(rvecs), np.array(tvecs), mtx, dist)
            imgpts = np.int32(imgpts).reshape(-1,2)

            cv2.drawContours(img, [imgpts[:4]], -1, color, thickness)
            cv2.drawContours(img, [imgpts[4:]], -1, color, thickness)
            for i in range(4):
                cv2.line(img, tuple(imgpts[i]), tuple(imgpts[i+4]), color, thickness)

            imgs.append(img.copy())

        for img in imgs:
            cv2.waitKey(500)
            cv2.imshow('Augmented Reality',img)

def my_hist(img, xlabel, ylabel, xlim):
    plt.clf()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.hist(img.ravel(), xlim[1], color='red', width=1.0)
    
    with io.BytesIO() as buf:
        plt.savefig(buf, format='png')
        buf.seek(0)
        return cv2.imdecode(np.frombuffer(buf.getbuffer(), np.uint8), -1)

def circle_detection(img):
    return cv2.HoughCircles(
        image = cv2.cvtColor(cv2.medianBlur(img, 5), cv2.COLOR_BGR2GRAY),
        method = cv2.HOUGH_GRADIENT,
        dp = 1.3,
        minDist = 30,
        param1 = 120,
        param2 = 30,
        minRadius = 0,
        maxRadius = 20)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
