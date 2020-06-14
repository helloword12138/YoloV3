#!/usr/bin/env python
# encoding: utf-8
"""
@version: 1.0
@author: liaoliwei
@contact: levio@pku.edu.cn
@file: people_flow.py
@time: 2018/7/9 14:52
"""


import colorsys
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
import cv2
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from keras.utils import multi_gpu_model

gpu_num=1

class YOLO(object):
    _defaults = {
        "model_path": 'logs/000/trained_weights_final.h5',  #已经训练好的模型
        "anchors_path": 'model_data/tiny_yolo_anchors.txt', # 通过聚类算法得到的anchor box
        "classes_path": 'model_data/coco_classes.txt', #可识别COCO类别列表
        "score" : 0.3, #框置信度阈值
        "iou" : 0.45, #IOU阈值， 大于阈值的重叠框会被删除
        "model_image_size" : (416, 416), #输入图片的大小
        "gpu_num" : 0,  #GPU的个数
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

#初始化
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

#完成目标检测
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # 读取 model路径 anchorbox，cococ类别 加载模型
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # 默认设置
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # 生成绘制边框的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # 固定种子为一致的进行运行
        np.random.shuffle(self.colors)  # 将颜色转换为相邻类
        np.random.seed(None)  # 将种子重置为默认

        # 为滤波后的包围盒生成输出张量目标
        self.input_image_shape = K.placeholder(shape=(2, ))
        if gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

#
    def detect_image1(self, image):
        # 开始计时
        start = timer()
        #调用letterbox_image()函数，先生成一个绝对灰R128-G128-B128，填充的新图片，然后用按比例缩放（采样方法：BICUBIC）后的输入图片粘贴，粘贴不到的部分保留为灰色
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            # model_image_size定义的宽和高必须是32的整倍数；若没有定义model_image_size，将输入图片的尺寸调整到32的整倍数，并调用letterbox_image()函数进行缩放
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        print(image_data.shape)
        image_data /= 255. #将缩放后图片的数值除以255，做归一化

        # 将（416,416,3）数组调整为（1,416,416,3）元组，满足YOLOv3输入的张量格式
        image_data = np.expand_dims(image_data, 0)

#生成结果  分别对三个feature map运行
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes], #将各种初始参数带入Tensorflow中运行
            feed_dict={  #输入参数
                self.yolo_model.input: image_data, #输入图片
                self.input_image_shape: [image.size[1], image.size[0]], #图片尺寸416x416
                K.learning_phase(): 0 #学习模式：0测试/1训练
            })
        lbox = []
        lscore = []
        lclass = []
        for i in range(len(out_classes)):
            if out_classes[i] == 0:
                lbox.append(out_boxes[i])
                lscore.append(out_scores[i])
                lclass.append(out_classes[i])
        out_boxes = np.array(lbox)
        out_scores = np.array(lscore)
        out_classes = np.array(lclass)
        print('画面中有{}个人'.format(len(out_boxes)))
        # 设置字体
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        # 设置目标框线条的宽度
        thickness = (image.size[0] + image.size[1]) // 300
        font_cn = ImageFont.truetype(font='font/asl.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        # 对于c个目标类别中的每个目标框i，调用Pillow画图
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]  #目标框
            score = out_scores[i]  #目标框的置信度评分
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)  #输出：绘制输入的原始图片
            label_size = draw.textsize(label, font)  #返回label的宽和高（多少个pixels）.
            top, left, bottom, right = box
            # 目标框的上、左两个坐标小数点后一位四舍五入
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            # 目标框的下、右两个坐标小数点后一位四舍五入，与图片的尺寸相比，取最小值
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))
            # 确定标签（label）起始点位置：左、下
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])
                # 画目标框，线条宽度为thickness
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            show_str = '  画面中有'+str(len(out_boxes))+'个人  '
			#标出图片中有多少个行人
            label_size1 = draw.textsize(show_str, font_cn)
            print(label_size1)
            # 画标签框
            draw.rectangle(
                [10, 10, 10 + label_size1[0], 10 + label_size1[1]],
                fill=(255,255,0))
            # 填写标签内容
            draw.text((10,10),show_str,fill=(0, 0, 0), font=font_cn)
            del draw
        end = timer()
        print(end - start)
        return image  #返回图片

    def close_session(self):
        self.sess.close()


def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        if(return_value == False):
            print("******************")
            break
        image = Image.fromarray(frame)
        image = yolo.detect_image1(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)

        if isOutput:
            out.write(result)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break


    vid.release()
    cv2.destroyAllWindows()


def detect_video1(yolo):
    vid = cv2.VideoCapture(0)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image = yolo.detect_image1(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()


def detect_img(yolo, img_path, output_path):
    image = Image.open(img_path)
    r_image = yolo.detect_image1(image)
    r_image.save(output_path)


