from flask import Flask, request,Response
import json
import ocr_model
from ocr_model import OCRModel
import time
import logging
from pub.logger import logger
from PIL import Image
import numpy as np

# 日志初始化
# logging.basicConfig(filename='logger.log', level=logging.INFO)
mylogger = logger(logger='TestMyLog').getlog()


# ctpn 模型初始化
mylogger.info("ctpn inti")
#actpn = ctpn.ctpn()
#actpn.init_model('model/ctpn.pb')
actpn = OCRModel('./ctpn/checkpoints', './ctpn/config/text.yml', 'densenet/models/weights_densenet.h5')

def generate_json(FilePath,boxes,errorcode,errormsg):

    response = {}
    response['file'] = FilePath
    response['errorcode'] = errorcode
    response['errormsg'] = errormsg
    response['items'] = []
    i=0
    print(boxes)

    for box in boxes:
        print(box)
        print(i)
        item = {}
        item['itemcoord'] = box[0:7]
        item['itemconf'] = box[8]
        response['items'].append(item)
        i = i + 1
    return response




app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'hello world'

@app.route('/ctpn', methods=['POST'])
def ctpn():

    ## 解析request
    TraceId = request.json['TraceId']
    FileList = request.json['FileList']
    Image1 = request.json['Image']
    js = json.dumps(request.json, sort_keys=True, indent=4, separators=(',', ':'))
    mylogger.info('request input : \n' + js)
    result_json = {}

    for i in FileList:
        mylogger.info('predict ' + i + ' begin')
        t = time.time()
        #boxes = actpn.predict(i)
        im = Image.open(i)
        img = np.array(im.convert('RGB'))
        scores, boxes, img, scale = actpn.ctpn(img)
        print(boxes)
        response = generate_json(i, boxes.tolist(), '0', 'OK')

        # result_json[i] = boxes.tolist()
        # print("Mission complete, it took {:.3f}s".format(time.time() - t))
        mylogger.info("Mission complete, it took {:.3f}s".format(time.time() - t))

    # print(Filelist)
    # print(result_json)
    # js = json.dumps(result_json, sort_keys=True, indent=4, separators=(',', ':'))
    js = json.dumps(response, sort_keys=True, indent=4, separators=(',', ':'))
    mylogger.info('request output : \n' + js)

    return Response(json.dumps(result_json), mimetype='application/json')

if __name__ == '__main__':
    app.run(port=5000, debug=True,use_reloader=False)
