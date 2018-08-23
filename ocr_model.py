import tensorflow as tf


import os
import cv2
from math import *
import numpy as np
from PIL import Image
from ctpn.ctpnlib.utils.timer import Timer
from ctpn.ctpnlib.fast_rcnn.config import cfg
from ctpn.ctpnlib.fast_rcnn.test import test_ctpn
from ctpn.ctpnlib.networks.factory import get_network
from ctpn.ctpnlib.text_connector.detectors import TextDetector
from ctpn.ctpnlib.text_connector.text_connect_cfg import Config as TextLineCfg
from ctpn.ctpnlib.fast_rcnn.config import cfg_from_file
graph = None

class OCRModel:
    def __init__(self, tf_model_path, conf_file, ks_model_path):
        cfg_from_file(conf_file)
        self.tf_model_path = tf_model_path
        self.net_name = 'VGGnet_test'
        self.sess, self.net = self.load_tf_model()
        from keras.layers import Input
        from keras.models import Model
        from densenet import keys
        from densenet import densenet
        self.characters = keys.alphabet[:]
        self.characters = self.characters[1:] + u'卍'
        self.nclass = len(self.characters)
        input = Input(shape=(32, None, 1), name='the_input')
        y_pred = densenet.dense_cnn(input, self.nclass)
        self.basemodel = Model(inputs=input, outputs=y_pred)
        modelPath = os.path.join(os.getcwd(), ks_model_path)
        if os.path.exists(modelPath):
            self.basemodel.load_weights(modelPath)
        global graph
        graph = tf.get_default_graph()

    def load_tf_model(self):
        # load config file
        cfg.TEST.checkpoints_path = self.tf_model_path

        # init session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        sess = tf.Session(config=config)

        # load network
        net = get_network(self.net_name)

        # load model
        print('Loading network {:s}... '.format(self.net_name))
        saver = tf.train.Saver()
        try:
            ckpt = tf.train.get_checkpoint_state(cfg.TEST.checkpoints_path)
            print('Restoring from {}...'.format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('done')
        except:
            raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)
        return sess, net

    def ctpn(self, img):
        timer = Timer()
        timer.tic()
        img, scale = self.resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
        scores, boxes = test_ctpn(self.sess, self.net, img)

        textdetector = TextDetector()
        boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
        timer.toc()
        print("\n----------------------------------------------")
        print(('Detection took {:.3f}s for '
               '{:d} object proposals').format(timer.total_time, boxes.shape[0]))
        return scores, boxes, img, scale

    def draw_boxes(self, img, boxes, scale):
        box_id = 0
        img = img.copy()
        text_recs = np.zeros((len(boxes), 8), np.int)
        for box in boxes:
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                continue
            if box[8] >= 0.8:
                color = (255, 0, 0)  # red
            else:
                color = (0, 255, 0)  # green
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
            cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)
            for i in range(8):
                text_recs[box_id, i] = box[i]

            box_id += 1
        cv2.imwrite("target.jpg", img)
        img = cv2.resize(img, None, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR)
        return text_recs, img

    def text_detect(self, img):
        scores, boxes, img, scale = self.ctpn(img)
        text_recs, img_drawed = self.draw_boxes(img, boxes, scale)
        return text_recs, img_drawed, img

    def resize_im(self, im, scale, max_scale=None):
        f = float(scale) / min(im.shape[0], im.shape[1])
        if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
            f = float(max_scale) / max(im.shape[0], im.shape[1])
        return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f

    def decode(self, pred):
        char_list = []
        pred_text = pred.argmax(axis=2)[0]
        for i in range(len(pred_text)):
            if pred_text[i] != self.nclass - 1 and (
                    (not (i > 0 and pred_text[i] == pred_text[i - 1])) or (i > 1 and pred_text[i] == pred_text[i - 2])):
                char_list.append(self.characters[pred_text[i]])
        return u''.join(char_list)

    def predict(self, img):
        width, height = img.size[0], img.size[1]
        scale = height * 1.0 / 32
        width = int(width / scale)
        img = img.resize([width, 32], Image.ANTIALIAS)
        '''
        img_array = np.array(img.convert('1'))
        boundary_array = np.concatenate((img_array[0, :], img_array[:, width - 1], img_array[31, :], img_array[:, 0]), axis=0)
        if np.median(boundary_array) == 0:  # 将黑底白字转换为白底黑字
            img = ImageOps.invert(img)
        '''
        img = np.array(img).astype(np.float32) / 255.0 - 0.5

        X = img.reshape([1, 32, width, 1])
        #print('x------------------------------------x')
        #print(X)
        self.basemodel._make_predict_function()
        y_pred = self.basemodel.predict(X)
        y_pred = y_pred[:, :, :]

        # out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1])[0][0])[:, :]
        # out = u''.join([characters[x] for x in out[0]])
        out = self.decode(y_pred)

        return out

    def charRec(self, img, text_recs, adjust=False):
        """
        加载OCR模型，进行字符识别
        """
        results = {}
        xDim, yDim = img.shape[1], img.shape[0]

        for index, rec in enumerate(text_recs):
            xlength = int((rec[6] - rec[0]) * 0.1)
            ylength = int((rec[7] - rec[1]) * 0.2)
            if adjust:
                pt1 = (max(1, rec[0] - xlength), max(1, rec[1] - ylength))
                pt2 = (rec[2], rec[3])
                pt3 = (min(rec[6] + xlength, xDim - 2), min(yDim - 2, rec[7] + ylength))
                pt4 = (rec[4], rec[5])
            else:
                pt1 = (max(1, rec[0]), max(1, rec[1]))
                pt2 = (rec[2], rec[3])
                pt3 = (min(rec[6], xDim - 2), min(yDim - 2, rec[7]))
                pt4 = (rec[4], rec[5])

            degree = degrees(atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))  # 图像倾斜角度

            partImg = self.dumpRotateImage(img, degree, pt1, pt2, pt3, pt4)

            if partImg.shape[0] < 1 or partImg.shape[1] < 1 or partImg.shape[0] > partImg.shape[1]:  # 过滤异常图片
                continue

            image = Image.fromarray(partImg).convert('L')
            text = self.predict(image)

            if len(text) > 0:
                results[index] = [rec]
                results[index].append(text)  # 识别文字

        return results

    def dumpRotateImage(self, img, degree, pt1, pt2, pt3, pt4):
        height, width = img.shape[:2]
        heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
        widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
        matRotation = cv2.getRotationMatrix2D((width // 2, height // 2), degree, 1)
        matRotation[0, 2] += (widthNew - width) // 2
        matRotation[1, 2] += (heightNew - height) // 2
        imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
        pt1 = list(pt1)
        pt3 = list(pt3)

        [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
        [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))
        ydim, xdim = imgRotation.shape[:2]
        imgOut = imgRotation[max(1, int(pt1[1])): min(ydim - 1, int(pt3[1])),
                 max(1, int(pt1[0])): min(xdim - 1, int(pt3[0]))]

        return imgOut

    def sort_box(self, box):
        """
        对box进行排序
        """
        box = sorted(box, key=lambda x: sum([x[1], x[3], x[5], x[7]]))
        return box

    def box_recs(self,boxes):
        box_id = 0
        text_recs = np.zeros((len(boxes), 9), np.float32)
        for box in boxes:
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                continue
            for i in range(9):
                text_recs[box_id, i] = box[i]
            box_id += 1
        text_recs = self.sort_box(text_recs)
        return text_recs

if __name__ == '__main__':
    model = OCRModel('./ctpn/checkpoints', './ctpn/config/text.yml', 'densenet/models/weights_densenet.h5')
    im = Image.open('./test_images/demo.jpg')
    img = np.array(im.convert('RGB'))
    result_dir = './test_result'
    scores, boxes, img, scale = model.ctpn(img)
    print(boxes)
    text_recs = model.box_recs(boxes)
    print(text_recs)
    result = model.charRec(img, text_recs, False)
    print(result)
    print("\nRecognition Result:\n")
    for key in result:
        print(result[key][1])
