import os
import cv2

def camera_auto_for_pictures(name):
    '''调用电脑摄像头自动获取图片'''
    save_dir = "D:\\CQU_learning\\Program_designing\\Python_program\\Facial_recognition\\data\Original_data\\"
    save = save_dir + name
    if not os.path.exists(save):
        os.makedirs(save)
    count = 1
    cap = cv2.VideoCapture(0)  # 打开摄像头
    # 设置摄像头参数
    width, height, w = 2000, 1500, 500
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # 设置摄像头宽度
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)  # 设置摄像头高度
    crop_w_start = (width - w) // 2
    crop_h_start = (height - w) // 2
    print("width:", width)
    print("height:", height)
    while True:
        # 开始截取照片
        ret, frame = cap.read()  # 截取到一帧照片，存放在frame中，ret为True
        frame = frame[crop_h_start:crop_h_start + w, crop_w_start:crop_w_start + w]
        frame = cv2.flip(frame, 1, dst=None)
        cv2.imshow("capture", frame)
        # 等待键盘操作
        action = cv2.waitKey(1) & 0xFF  # 窗口出现10s内输入一个值
        if action == ord("c"):  # 按键c，表示创建新文件夹，用于存储截取的图片
            name = input("请输入新的存储文件夹名称:")
            save = save_dir + name
            if not os.path.exists(save):
                os.makedirs(save)
        elif action == ord("p"):  # 按键p，将图片按指定名称保存到指定路径


            cv2.imwrite(u"%s\\%s.jpg" % (save, name + str(count)), cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA))
            print(u"%s: %d张图片" % (save, count))
            count += 1
        elif action == ord("q"):  # 按键q，表示推出拍摄
            break
    cap.release()  # 停止捕获摄像头
    cv2.destroyAllWindows()  # 关闭相应的显示窗口

if __name__ == "__main__":
    camera_auto_for_pictures(name="Heyuxiang")