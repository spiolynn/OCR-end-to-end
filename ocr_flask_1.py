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

def generate_json_v2(FilePath,result,errorcode,errormsg,TraceId):
    response = {}
    response['TraceId'] = TraceId
    response['File'] = FilePath
    response['ErrorCode'] = errorcode
    response['ErrorMsg'] = errormsg
    response['Items'] = []
    i = 0

    for key,value in result.items():
        box_id = key
        box_item = value
        item = {}
        item['ItemCoord'] = value[0].tolist()[0:8]
        item['ItemConf'] = value[0].tolist()[8]
        item['Words'] = value[1]
        response['Items'].append(item)
        i = i + 1
    return response

def generate_json(FilePath,boxes,errorcode,errormsg,TraceId):

    response = {}
    response['file'] = FilePath
    response['errorcode'] = errorcode
    response['errormsg'] = errormsg
    response['TraceId'] = TraceId
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
    j = 0

    # for i in FileList:
    i = FileList    
    mylogger.info('predict ' + i + ' begin')
    t = time.time()
        #boxes = actpn.predict(i)
    try:
        im = Image.open(i)
        img = np.array(im.convert('RGB'))
        scores, boxes, img, scale = actpn.ctpn(img)
        #print(boxes)
        #print(type(scores))
        #print(scores)
        #print(img)
        #print(type(img))
        #print(type(scale))
        #print(scale)
        # response = generate_json(i, boxes.tolist(), '0', 'OK')
        # result_json[i] = boxes.tolist()

        text_recs = actpn.box_recs(boxes)
        # print(text_recs)
        # print(type(text_recs))
        with ocr_model.graph.as_default(): 
            result = actpn.charRec(img, text_recs, False)        
        response = generate_json_v2(i,result,'0','OK',TraceId)
        #print("Mission complete, it took {:.3f}s".format(time.time() - t))
        mylogger.info("Mission complete, it took {:.3f}s".format(time.time() - t))

        # print(Filelist)
        # print(result_json)
        # js = json.dumps(result_json, sort_keys=True, indent=4, separators=(',', ':'))
        js = json.dumps(response, sort_keys=True, indent=4, separators=(',', ':'),ensure_ascii=False)
        mylogger.info('request output : \n' + js)

        return Response(json.dumps(js), mimetype='application/json')
    except Exception as e:
        mylogger.info('error : \n' + str(e))
        response = generate_json_v2(i,{},'1',str(e),TraceId)
        js = json.dumps(response, sort_keys=True, indent=4, separators=(',', ':'),ensure_ascii=False)
        mylogger.info('request output : \n' + js)
        return Response(json.dumps(js), mimetype='application/json')
    
if __name__ == '__main__':
    app.run(port=5000, debug=True,use_reloader=False)
