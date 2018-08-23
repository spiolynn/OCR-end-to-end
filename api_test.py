import requests
import json

# user_info = {'name': ['letian', 'letian2'], 'password': '123'}

def test1():

    '''
    模拟CTPN API 接口测试
    :return:
    '''

    # orc_info = {'FilePath': ['1.jpg', '2.jpg'], 'FileId': '001'}
    # r = requests.post("http://127.0.0.1:5000/ctpn", data=orc_info)
    orc_info = {'TraceId': '20180808010101-0000001','FileList': './test_images/demo1.jpg', 'Image': ''}
    r = requests.post("http://127.0.0.1:5000/ctpn", json=orc_info)
    js = json.loads(r.text)
    print(js)

if __name__ == '__main__':
    test1()

