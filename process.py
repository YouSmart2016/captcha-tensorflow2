import  cv2 as  cv

import  numpy as np
import  os
import glob
import  tensorflow as tf

from  PIL import Image



imagePath='./images/char-4-epoch-3/train/0125_df6fc8ea-7e74-4cd3-99b9-84fa2025b118.png'
while_blackimagePath='./white_black/images/char-4-epoch-1/train/0123_a701419d-7c3d-4cca-b8fa-adb4ae613fbb.png'
'''
path='./images/char-4-epoch-3/test/'
files = glob.glob(os.path.join(path, "*.png"))

for filePath in files:
    image=cv.imread(filePath)
    grayImage=cv.cvtColor(image,cv.COLOR_BGR2GRAY)

    blur=cv.medianBlur(grayImage,5)
    # blur = cv.GaussianBlur(grayImage,(5,5),0)

    retval,binImg=cv.threshold(blur,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)

    # kernel=cv.getStructuringElement(cv.MORPH_RECT,(5,5))

    # res = cv.morphologyEx(binImg,cv.MORPH_OPEN,kernel=kernel)
    blur=cv.medianBlur(binImg,5)
    image=np.hstack((grayImage,binImg))
    cv.imshow('hstack',image)
    cv.waitKey()
    
    '''


def preprocess(imagePath):
    image=cv.imread(imagePath)
    grayImage=cv.cvtColor(image,cv.COLOR_BGR2GRAY)

    blur=cv.medianBlur(grayImage,5)
    # blur = cv.GaussianBlur(grayImage,(5,5),0)

    retval,binImg=cv.threshold(blur,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)
    binImg = cv.merge((binImg, binImg, binImg))  # 合并三通道
    binImg=cv.bitwise_not(binImg)
    # binImg=binImg[...,tf.newaxis]
    return binImg


res=preprocess(imagePath)
cv.imshow('code',res)
cv.waitKey()
print(res.shape)

'''
image = cv.imread(imagePath)
grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
retval, binImg = cv.threshold(grayImage, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
pic=cv.merge((binImg,binImg,binImg)) #合并三通道

res1=np.array(pic)
cv.imshow('code',res1)
cv.waitKey()
print(res1.shape)
'''
# image_file.save('result.png')