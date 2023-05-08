import os
import shutil
import glob
import cv2

'''
i = 0
folder_path = '/home/dungdinh/Documents/Prj2/line_4'

for file in glob.glob('/home/dungdinh/Documents/Prj2/line/*' ):
    if i<2000:
        shutil.move(file, folder_path )
        i +=1
'''
img = cv2.imread('/home/dungdinh/Documents/Prj2/line_dam/0d76d7f9654743768494f28c74c98eaf_0_.jpg')
print(img.shape)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', img_gray)
cv2.waitKey(0)