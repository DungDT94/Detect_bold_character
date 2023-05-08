import os
import uuid
folder_path = "/home/dungdinh/Documents/Prj2/data/data_train/1/"
file_jpg = [f for f in os.listdir(folder_path) if ".jpg" in f]
for file in file_jpg:
    absolute_file_path_jpg = folder_path + file
    name_uuid = uuid.uuid4().hex
    name_jpg = name_uuid + '_1_' + ".jpg"
    name_txt = name_uuid + ".txt"
    absolute_file_path_jpg_uuid = folder_path + name_jpg
    os.rename(absolute_file_path_jpg, absolute_file_path_jpg_uuid)
