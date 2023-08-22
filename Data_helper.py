import os
import cv2
import time

def read_all_img(path, *suffix):
    '''基于后缀读取文件'''
    try:
        s = os.listdir(path)  # 返回文件夹中包含文件民资的列表
        result_array = []
        filename = os.path.basename(path)  # 返回路径名的基名
        result_array.append(filename)
        for i in s:  # 遍历s中的每个文件名
            if endwith(i, suffix):  # 判断i是否以suffix结尾
                document = os.path.join(path, i)  # 将路径path与文件名i组合得到新路径，指向对应的待处理图片文件
                img = cv2.imread(document)  # 返回文件的相关数据,(高度，宽度，通道数)元组
                result_array.append(img)
    except IOError:
        print("Error")
    else:
        print("读取成功")
        return result_array

def endwith(s, *endstring):
    '''对字符串的后缀进行匹配'''
    result_array = map(s.endswith, endstring)  # 判断s是否以endstring序列中的任意字符串结尾
    if True in result_array:
        return True
    else:
        return False

def read_pic_save_face(source_path, dst_path, *suffix):
    '''图片标准化与存储'''
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    try:
        result_array = read_all_img(source_path, *suffix)  # 基于后缀读取文件
        count = 1
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
        for i in result_array:
            if type(i) != str:
                gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)  # 将图片i转化为灰度图
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # 进行图片中的人脸检测
                for(x, y, w, h) in faces:  # 每个人脸的坐标值，宽度和高度
                    liststr = [str(int(time.time())), str(count)]
                    filename = "".join(liststr)  # 以""为分隔符将liststr中的每个元素组合成一个字符串
                    f = cv2.resize(gray[y : (y + h), x : (x + w)], (200, 200))
                    cv2.imwrite(dst_path + os.sep + "%s.jpg" % filename, f)
                    count += 1
    except Exception as e:
        print("Exception；", e)
    else:
        print("Read " + str(count - 1) + " Faces to Destination " + dst_path)

if __name__ == "__main__":
    print("dataProcessing!!!")
    path_s = "D:\\CQU_learning\\Program_designing\\Python_program\\Facial_recognition\\data\\Original_data"
    path_d = "D:\\CQU_learning\\Program_designing\\Python_program\\Facial_recognition\\data\\Processed_data"
    dir_name = os.listdir(path_s)
    for child_dir in dir_name:
        source_path = path_s + "\\" + child_dir
        dst_path = path_d + "\\" + child_dir
        read_pic_save_face(source_path, dst_path, ".jpg", ".JPG", ".png", ".PNG", ".tiff")