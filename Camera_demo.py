import os
import cv2
from Face_recognition_model import Model
from time import sleep

threshold = 0.7

def read_name_list(path):  # 返回处理后的数据文件夹中的每个子文件夹名
    '''读取训练数据集'''
    name_list = []
    for child_dir in os.listdir(path):
        name_list.append(child_dir)
    return name_list


class Camera_reader(object):
    def __init__(self):
        self.model = Model()
        self.model.load()  # 加载训练完成后的"face.h5"文件
        self.img_size = 128

    def build_camera(self):
        '''调用摄像头进行实时人脸识别'''
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")  # 构建人脸识别分类器
        name_list = read_name_list(
            "D:\\CQU_learning\\Program_designing\\Python_program\\Facial_recognition\\data\\Processed_data")
        cap = cv2.VideoCapture(0)  # 捕获摄像头
        success, frame = cap.read()  # 截取一帧图片
        while success and cv2.waitKey(1) == -1:
            success, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将图片转换为灰度图
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # 进行检测
            for (x, y, w, h) in faces:
                ROI = gray[x:x + w, y:y + h]
                try:
                    ROI = cv2.resize(ROI, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                except:
                    continue
                label, prob = self.model.predict_one(ROI)
                if prob > threshold:
                    show_name = name_list[label]
                else:
                    show_name = "Stranger"
                cv2.putText(frame, show_name, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)  # 给图片添加文字
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 在图片上绘制矩形框
            cv2.imshow("Camera", frame)
        else:  # 当while循环正常结束时会执行该else代码块
            cap.release()
            cv2.destroyAllWindows()
            return show_name

if __name__ == "__main__":
    camera = Camera_reader()
    camera.build_camera()