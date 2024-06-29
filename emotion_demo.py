from __future__ import print_function
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont
from deepface import DeepFace


def get_video_info(video_cap):

    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    numFrames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    return width, height, numFrames, fps


def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

cap = cv2.VideoCapture(0)


while True:
    isuccess, frame = cap.read()
    if not isuccess:
        print("No more frames")
        break
    # 分离左右摄像头
    # imgL = frame[:, :int(width / 2), :]
    img = frame
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        img_w = int(img.shape[1] / 2)
        img_h = int(img.shape[0] / 2)
        demography = DeepFace.analyze(img,actions=["emotion", "age", "gender", "race"])
        emotion = demography[0]['dominant_emotion']
        rate = demography[0]['emotion'][emotion]
        region = demography[0]['region']
        age = demography[0]['age']
        gender = demography[0]['dominant_gender']
        race = demography[0]['dominant_race']

        
        img = cv2.rectangle(img,(region['x'],region['y']),(region['x']+region['w'],region['y']+region['h']),(0,255,0),2)
        img = cv2AddChineseText(img, '{}-{}-{}-{}'.format(emotion,age,gender,race), (region['x'], region['y']-60), textColor=(0, 255, 0), textSize=60)
        # pass
        img = cv2AddChineseText(img,f'{fps}-{img_w}-{img_h}', (0, 0), textColor=(0, 255, 0), textSize=10)
    except ValueError:
        print('Face could not be detected. Please confirm that the picture is a face photo or consider to set enforce_detection param to False.')
        img = frame
        # Displaying the disparity map
        # 显示结果
    cv2.imshow("left image",img)
    # cv2.imshow("right image",imgR)
    key = cv2.waitKey(10)
    if key == ord('q'):
        break
    
cv2.destroyAllWindows()
