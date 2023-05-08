import os
import cv2
import numpy as np
import math
import glob

def preprocess(image):
    image = cv2.imread(image)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height = img_gray.shape[0]
    width = img_gray.shape[1]
    number_box = int(width / height)  # so box trong chieu cao bang chieu rong
    line_number = int(math.sqrt(number_box) + 0.7)  # lay binh phuong cua so box
    if line_number == 0:
        return image
    else:
        unit = int(width / (line_number))
        img = img_gray[:, 0: unit * (0 + 1) + int(unit / 8)]
        #cv2.imshow('crop 1', img)
        #cv2.waitKey(0)
        for i in range(1, line_number):
            if i == (line_number - 1):
                img_temp = img_gray[:, i * unit: unit * (i + 1)]
                img_add = np.ones((height, int(unit / 8)), dtype=np.uint8)
                img_add = img_add * 255
                img_temp_2 = cv2.hconcat([img_temp, img_add])
                #cv2.imshow('img_temp', img_temp)
                #cv2.waitKey(0)
                img = cv2.vconcat([img, img_temp_2])
            else:
                img_temp = img_gray[:, i * unit: unit * (i + 1) + int(unit / 8)]
                #cv2.imshow('img_temp', img_temp)
                #cv2.waitKey(0)
                img = cv2.vconcat([img, img_temp])

        #height = img.shape[0]
        #width = img.shape[1]
        #cv2.imshow('img_thua', img)
        #print(img.shape)
        #cv2.waitKey(0)
        img_add_h = np.ones((img.shape[0], int(unit / 8)), dtype=np.uint8)
        img_add_h = img_add_h * 255
        img = cv2.hconcat([img, img_add_h])
        img = cv2.hconcat([img_add_h, img])
        img_add_v = np.ones((int(unit / 8), img.shape[1]), dtype=np.uint8)
        img_add_v = img_add_v * 255
        img = cv2.vconcat([img, img_add_v])
        img = cv2.vconcat([img_add_v, img])

        return img


def preprocess_pad(image):
    image = cv2.imread(image)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height = img_gray.shape[0]
    width = img_gray.shape[1]
    number_box = int(width / height)  # so box trong chieu cao bang chieu rong
    line_number = int(math.sqrt(number_box) + 0.7)  # lay binh phuong cua so box
    if line_number == 0:
        return image
    else:
        unit = int(width / (line_number))
        img = img_gray[:, 0: unit * (0 + 1)]
        #cv2.imshow('crop 1', img)
        #cv2.waitKey(0)
        for i in range(1, line_number):
            img_temp = img_gray[:, i * unit: unit * (i + 1)]
            #cv2.imshow('img_temp', img_temp)
            #cv2.waitKey(0)
            img = cv2.vconcat([img, img_temp])

        #height = img.shape[0]
        #width = img.shape[1]
        #cv2.imshow('img_thua', img)
        #print(img.shape)
        #cv2.waitKey(0)
        img_add_h = np.ones((img.shape[0], int(unit/5)), dtype=np.uint8)
        img_add_h = img_add_h * 255
        img = cv2.hconcat([img, img_add_h])
        img = cv2.hconcat([img_add_h, img])
        img_add_v = np.ones((int(unit/5), img.shape[1]), dtype=np.uint8)
        img_add_v = img_add_v * 255
        img = cv2.vconcat([img, img_add_v])
        img = cv2.vconcat([img_add_v, img])

        return img

def preprocess_rec(image):
    image = cv2.imread(image)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height = img_gray.shape[0]
    width = img_gray.shape[1]
    number_box = int(width / height)  # so box trong chieu cao bang chieu rong
    line_number = int(math.sqrt(number_box) + 0.7)  # lay binh phuong cua so box
    if line_number == 0:
        return image
    else:
        unit = int(width / (line_number))
        img = img_gray[:, 0: unit * (0 + 1)]
        # cv2.imshow('crop 1', img)
        # cv2.waitKey(0)
        for i in range(1, line_number):
            img_temp = img_gray[:, i * unit: unit * (i + 1)]
            # cv2.imshow('img_temp', img_temp)
            # cv2.waitKey(0)
            img = cv2.vconcat([img, img_temp])

        # height = img.shape[0]
        # width = img.shape[1]
        # cv2.imshow('img_thua', img)
        # print(img.shape)
        # cv2.waitKey(0)
        print('shape truoc', img.shape)
        if img.shape[0] > img.shape[1]:
            minus = img.shape[0] - img.shape[1]
            pad = int(minus/2)
            print('pad', pad)
            img_add_h = np.ones((img.shape[0], pad), dtype=np.uint8)
            img_add_h = img_add_h * 255
            img = cv2.hconcat([img, img_add_h])
            img = cv2.hconcat([img_add_h, img])
        elif img.shape[0] < img.shape[1]:
            minus = img.shape[1] - img.shape[0]
            pad = int(minus/2)
            print('pad', pad)
            img_add_v = np.ones((pad, img.shape[1]), dtype=np.uint8)
            img_add_v = img_add_v * 255
            img = cv2.vconcat([img, img_add_v])
            img = cv2.vconcat([img_add_v, img])
        else :
            pass
        print('shape sau', img.shape)
        return img


def write_folder(folder_path,folder_save):
    files = [file for file in glob.glob(folder_path + '/*.jpg')]
    for file in files:
        print(file)
        base_name = os.path.basename(file)
        print(base_name)
        img = preprocess(file)
        file_path_save = os.path.join(folder_save, base_name)
        print(file_path_save)
        cv2.imwrite(file_path_save, img)


if __name__ == "__main__":
    write_folder('/home/dungdinh/Documents/Prj2/data/line_1_done', '/home/dungdinh/Documents/Prj2/data/data_train/line_nhat')
    #preprocess('/home/dungdinh/Documents/Prj2/data/line_dam/0a4a3e4f978248fe951f95c87786429a_0_.jpg')




