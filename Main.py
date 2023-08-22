from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask import redirect, url_for, render_template
import os
import cv2
import time
import numpy as np
from flask import request
from Face_recognition_model import Model
from Camera_demo import Camera_reader
from flask import Flask
from flask_sqlalchemy import SQLAlchemy  # 要用python操作数据库，必须导入这个包
import time

app = Flask(__name__)
app.config['UPLOADED_PHOTO_DEST'] = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOADED_PHOTO_ALLOW'] = IMAGES
photos = UploadSet('PHOTO')
configure_uploads(app, photos)

HOSTNAME = "127.0.0.1"
PORT = 3306
USERNAME = "root"
PASSWORD = ""
DATABASE = "user_info"
app.config['SQLALCHEMY_DATABASE_URI'] = f"mysql+pymysql://{USERNAME}:{PASSWORD}@{HOSTNAME}:{PORT}/{DATABASE}?charset=utf8mb4"
db = SQLAlchemy(app)


class User(db.Model):
    __tablename__ = "user"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    realname = db.Column(db.String(100), nullable=False)
    position = db.Column(db.String(100), nullable=False)
    username = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)
    check_times = db.Column(db.Integer)
    true_check_times = db.Column(db.Integer)
    false_check_times = db.Column(db.Integer)
    no_check_times = db.Column(db.Integer)


with app.app_context():
    db.create_all()


def dest(name):
    return "{}/{}".format(app.config.UPLOADED_PHOTO_DEST, name)


def detect_one_picture(path):
    '''单图识别'''
    model = Model()
    model.load()
    img = cv2.imread(path)
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    picType, prob = model.predict_one(img)
    if picType != -1:
        name_list = read_name_list(
            "D:\\CQU_learning\\Program_designing\\Python_program\\Facial_recognition\\data\\Processed_data")
        print(name_list[picType], prob)
        res = u"识别为：" + name_list[picType] + u"的概率为：" + str(prob)
        print(res)
    else:
        res = u"抱歉，未识别出该人！请尝试增加数据量来训练模型！"
    return res


def endwith(s, *endstring):
    result_array = map(s.endswith, endstring)
    if True in result_array:
        return True
    else:
        return False


def read_file(path):
    img_list = []
    label_list = []
    dir_counter = 0
    IMG_SIZE = 128
    for child_dir in os.listdir(path):
        child_path = os.path.join(path, child_dir)
        for dir_image in os.listdir(child_path):
            if endwith(dir_image, '.jpg'):
                img = cv2.imread(os.path.join(child_path, dir_image))
                resized_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                recolored_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
                img_list.append(recolored_img)
                label_list.append(dir_counter)
        dir_counter += 1
    img_list = np.array(img_list)
    return img_list, label_list, dir_counter


def read_name_list(path):
    name_list = []
    for child_dir in os.listdir(path):
        name_list.append(child_dir)
    return name_list


@app.route("/", methods=['POST', 'GET'])
def init():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        realname = request.form.get("realname")
        position = request.form.get("position")
        user = User(realname=realname, password=password, username=username, position=position, check_times=0,
                    true_check_times=0, false_check_times=0, no_check_times=0)
        db.session.add(user)
        db.session.commit()

    content = []
    for user in User.query.all():
        item = [user.id, user.realname, user.position, user.check_times]
        content.append(item)
    return render_template("index.html", content=content)


@app.route("/check")
def check():
    return render_template('check.html', title='Home')


@app.route("/she")
def she():
    camera = Camera_reader()
    name = camera.build_camera()
    user = User.query.filter_by(username=name).first()

    t = time.localtime()  # 获取当前时间
    if t.tm_hour < 8 or (t.tm_hour == 17 and t.tm_min < 30):
        if user.check_times < 2:
            user.check_times += 1
        if user.true_check_times < 2:
            user.true_check_times += 1
    elif (t.tm_hour == 8 and t.tm_min < 30) or 12 <= t.tm_hour < 17:
        if user.check_times < 2:
            user.check_times += 1
        if user.false_check_times < 2:
            user.false_check_times += 1
    elif (t.tm_hour == 8 and t.tm_min >= 30) or (8 < t.tm_hour < 12) or (t.tm_hour == 17 and t.tm_min >= 30) or (t.tm_hour > 17):
        if user.no_check_times < 2:
            user.no_check_times += 1

    db.session.commit()
    return render_template("check.html", title='Home')


@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        return redirect(url_for('show', name=filename))
    return render_template('upload.html')


@app.route('/photo/<name>')
def show(name):
    if name is None:
        print("出错了！")
    url = photos.url(name)
    if request.method == 'GET':
        picture = name
    start_time = time.time()
    res = detect_one_picture(picture)
    end_time = time.time()
    execute_time = str(round(end_time - start_time, 2))
    tsg = u'总耗时为：%s秒' % execute_time
    return render_template('show.html', url=url, name=name, xinxi=res, shijian=tsg)


# 制作注册与登录的界面
@app.route("/login")
def login():
    return render_template("login.html")


@app.route("/register")
def register():
    return render_template("register.html")


@app.route("/logged", methods=['GET', 'POST'])
def logged():
    username = request.form.get("username")
    password = request.form.get("password")
    for user in User.query.all():
        if (user.username == username) and (user.password == password):
            content = []
            for user in User.query.all():
                item = [user.id, user.realname, user.position, user.check_times]
                content.append(item)
            return render_template("logged.html", username=username, content=content)
    return render_template("unlogged.html")


@app.route("/information/<username>")
def information(username):
    user = User.query.filter_by(username=username).first()
    content = [user.check_times, user.true_check_times, user.false_check_times, user.no_check_times]
    return render_template("information.html", username=username, content=content)


if __name__ == "__main__":
    print("FaceRecognitionDemo")
    app.run(debug=True)
