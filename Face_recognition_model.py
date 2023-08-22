import os
import cv2
import random
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten

def endwith(s, *endstring):
    '''
    对字符串的后续和标记进行匹配（尾部匹配）
    '''
    result_array = map(s.endswith, endstring)
    if True in result_array:
        return True
    else:
        return False

def read_file(path):  # 读取指定路径的图片信息
    img_list = []  # 存储图片数据的列表
    label_list = []  # 存储图片对应的标签（下标）的列表
    dir_counter = 0
    IMG_SIZE = 128  # 图片大小设置
    for child_dir in os.listdir(path):
        child_path = os.path.join(path, child_dir)
        for dir_image in os.listdir(child_path):
            if endwith(dir_image, ".jpg"):
                img = cv2.imread(os.path.join(child_path, dir_image))  # 读取预处理后的图像信息
                resized_image = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # 缩放图像的大小
                recolored_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)  # 色彩空间的转换
                img_list.append(recolored_image)  # 将当前图片信息作为一个元素插入
                label_list.append(dir_counter)  # 插入当前图片对应的下标（索引值）
        dir_counter += 1
    img_list = np.array(img_list)  # 将图片数据列表转换为数组
    return img_list, label_list, dir_counter

class DataSet(object):
    def __init__(self, path):
        '''初始化'''
        self.num_classes = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.img_size = 128
        self.extract_data(path)

    def extract_data(self, path):
        imgs, labels, counter = read_file(path)
        x_train, x_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.2, random_state=random.randint(0, 100))
        #  将原始数据集按照1：4划分成测试集与训练集
        x_train = x_train.reshape(x_train.shape[0], 1, self.img_size, self.img_size) / 255.0
        # 在不更改数据的情况下为数组赋予新形状,将图片转换为指定的尺寸和灰度
        x_test = x_test.reshape(x_test.shape[0], 1, self.img_size, self.img_size) / 255.0
        x_train = x_train.astype('float32')  # 设置图片数据类型为“float32”
        x_test = x_test.astype('float32')
        y_train = np_utils.to_categorical(y_train, num_classes=counter)  # 将标签转换为one-hot编码
        y_test = np_utils.to_categorical(y_test, num_classes=counter)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.num_classes = counter

    def check(self):
        '''校验'''
        print("num of dim:", self.x_test.ndim)  # 返回数组的维度
        print("shape:", self.x_test.shape)  # 返回数组的形状：各维度大小的元组
        print("size:", self.x_test.size)  # 返回数组的大小：元素个数
        print("num of dim:", self.x_train.ndim)
        print("shape:", self.x_train.shape)
        print("size:", self.x_train.size)

class Model(object):
    '''人脸识别模型'''
    FILE_PATH = "face.h5"
    IMG_SIZE = 128

    def __init__(self):
        self.model = None

    def read_train_data(self, dataset):
        self.dataset = dataset

    def build_model(self):
        self.model = Sequential()  # 先生成一个空容器，容纳神经网络的网络结构
        self.model.add(  # 通过add方法将各层添加到模型中
            Convolution2D(  # 添加第一个卷积层
                filters=32,  # 过滤器、卷积核个数，或者卷积后的输出通道数
                kernel_size=(5, 5),  # 卷积核尺寸5*5
                padding="same",  # 边缘用0填充
                # dim_ordering="th",  # 设置图像维度顺序（通道维放在第二个位置）
                input_shape=self.dataset.x_train.shape[1:]
            )
        )
        self.model.add(Activation("relu"))
        self.model.add(
            MaxPooling2D(  # 添加第一个池化层
                pool_size=(2, 2),  # 设置池化层尺寸2*2
                strides=(2, 2),  # 移动步长
                padding="same",  # 边缘用0填充
            )
        )
        self.model.add(Convolution2D(filters=64, kernel_size=(5, 5), padding="same"))  # 添加第二个卷积层
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))  # 添加第二个池化层
        self.model.add(Flatten())  # 把多维输入一维化，常用在从卷积层到全连接层的过渡
        self.model.add(Dense(1024))  # 添加第一个全连接层
        self.model.add(Activation("relu"))
        self.model.add(Dense(self.dataset.num_classes))  # 添加第二个全连接层
        self.model.add(Activation("softmax"))
        self.model.summary()  # 打印显示网络结构和参数

    def train_model(self):
        self.model.compile(
            optimizer="adam",  # 优化器，用于控制梯度裁剪
            loss="categorical_crossentropy",  # 损失函数
            metrics=["accuracy"]  # 评价函数，用于评估当前训练模型的性能
        )
        self.model.fit(self.dataset.x_train, self.dataset.y_train, epochs=10, batch_size=10)
        # 在模型中训练一定次数，epochs指定迭代次数为10，batch_size指定训练一次网络所用的样本数为10

    def evaluate_model(self):
        print("\nTesting------------")
        loss, accuracy = self.model.evaluate(self.dataset.x_test, self.dataset.y_test)
        print("test loss:", loss)
        print("test accuracy:", accuracy)

    def save(self, file_path=FILE_PATH):
        print("Model Saved Finished!!!")
        self.model.save(file_path)  # 保存训练完成的模型

    def load(self, file_path=FILE_PATH):
        print("Model Loaded Successful!!!")
        self.model = load_model(file_path)  # 加载模型

    def predict_one(self, img):
        img = img.reshape((1, 1, self.IMG_SIZE, self.IMG_SIZE))
        img = img.astype("float32")
        img = img / 255.0
        result = self.model.predict(img)  # 返回n行k列数组，i行j列为模型预测第i个样本为某个标签的概率
        max_index = np.argmax(result)  # 返回数组最大值的索引值
        return max_index, result[0][max_index]

if __name__ == "__main__":
    path = "D:\\CQU_learning\\Program_designing\\Python_program\\Facial_recognition\\data\\Processed_data"
    dataset = DataSet(path)
    model = Model()
    model.read_train_data(dataset)
    model.build_model()
    model.train_model()
    model.evaluate_model()
    model.save()