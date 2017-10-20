

import imutils
import numpy as np
import cv2
import scipy.misc




##plate_cascade = cv2.CascadeClassifier('li.xml')
# cap = cv2.VideoCapture('v2.mp4')
#
# while(cap.isOpened()):
#     ret, img = cap.read()
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     plate = plate_cascade.detectMultiScale(gray, 1.3, 5)
#
#     for (x, y, w, h) in plate:
#         cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         im2 = img[y: y + h, x: x + w]
#         cv2.imwrite('im2.jpg',im2)
#
#
#     cv2.imshow('img', img)
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break
#
# cap.release()
# cv2.destroyAllWindows()


img = cv2.imread('im2.jpg')
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
th = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

thresh = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
# thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
digitCnts = []

for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    print(x,y,w,h)
    img = img[y: y + h, x: x + w]
    cv2.imshow('boxedImage', img)
    if w >= 15 and (30 <= h <= 40):
        digitCnts.append(c)

# print(digitCnts,cnts)
# cv2.imshow('boxedImage', img)

grayscale=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
th = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

thresh = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
# thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
digitCnts = []
digits=[]

for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    # print(x,y,w,h)
    if w >= 15 and (30 <= h <= 40):

        digitCnts.append(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        mask = np.zeros((128,400,3))
        mask[:] = (255,255,255)
        cv2.drawContours(mask,[c],0,(0,0,0),-1)
        out = np.zeros_like(img)
        cv2.imshow('test',mask)
        # cv2.imwrite('files{f}.png'.format(f = c),mask)
        # out[mask == 255] = img[mask == 255]
        #  mask = mask[ x:x+w+20, y-10:y+h+20]
        # cv2.imshow('Output',out)
        digits.append(mask)
        # print("imagee",image2)

cv2.imshow("contoured", img)
print(digits)

for image in digits:
    z = 1
    cv2.imshow('detect{f}.jpg'.format(f=z), image)
    z = z+1
    # scipy.misc.toimage(image, cmin=0.0, cmax=...).save('detect{f}.jpg'.format(f=z))

# print("digits",digits)
# print("reshape", digits[0].reshape([-1,28,28,1]))
# print(digitCnts,cnts)
# cv2.imshow('boxedImage', img)
#cv2.imshow('original',img)
cv2.imshow('threshold',thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()