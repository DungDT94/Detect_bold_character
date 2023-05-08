import json
import cv2
import uuid
import glob


def crop(image_path):
    json_path = image_path.split('.')[0] + '.json'
    f = open(json_path)
    image = cv2.imread(image_path)
    data = json.load(f)
    for line in data['result']:
        line_lst = []
        # print('line:', line)
        for subline in line:
            # print('subline:', subline)
            subline_lst = []
            subline_lst_coor = []
            for box in subline:
                # print('box:', box)
                box_lst = box['box']
                box_crop = image[int(box_lst[1]):int(box_lst[3]), int(box_lst[0]):int(box_lst[2])]
                # cv2.imwrite('crop_1/' + uuid.uuid4().hex + '_0_' + '.jpg', box_crop)
                subline_lst.append(box_lst)
            if len(subline_lst) != 0:
                subline_xmin = sorted(subline_lst, key=lambda x: x[0])[0][0]
                subline_ymin = sorted(subline_lst, key=lambda x: x[1])[0][1]
                subline_xmax = sorted(subline_lst, key=lambda x: x[2], reverse=True)[0][2]
                subline_ymax = sorted(subline_lst, key=lambda x: x[3], reverse=True)[0][3]
                subline_lst_coor.extend([subline_xmin, subline_ymin, subline_xmax, subline_ymax])
                subline_crop = image[subline_ymin:subline_ymax, subline_xmin:subline_xmax]
                line_lst.append(subline_lst_coor)
            if len(subline_lst) != 1 and len(subline_lst) != 0:
                cv2.imwrite('line/' + uuid.uuid4().hex + '_0_' + '.jpg', subline_crop)
        #print('line', line_lst)
        if  len(line_lst) != 0:
            line_xmin = sorted(line_lst, key=lambda x: x[0])[0][0]
            line_ymin = sorted(line_lst, key=lambda x: x[1])[0][1]
            line_xmax = sorted(line_lst, key=lambda x: x[2], reverse=True)[0][2]
            line_ymax = sorted(line_lst, key=lambda x: x[3], reverse=True)[0][3]
            line_crop = image[line_ymin:line_ymax, line_xmin:line_xmax]
        if len(line_lst) != 1 and len(line_lst) != 0:
            cv2.imwrite('line/' + uuid.uuid4().hex + '_0_' + '.jpg', line_crop)


if __name__ == "__main__":
    folder = '/home/dungdinh/Documents/Prj2/json_docx (copy)'
    files = glob.glob(folder + '/*.jpg')
    for file in files:
        crop(file)
        #print(file)

