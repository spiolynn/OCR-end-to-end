# OCR-end-to-end


## ORC服务发布

[toc]

---
文档版本 | 文档更新时间 
---|---
V0.1 | 2018-08-22 
---

### 1 环境要求

#### 1 Linux

- python 3.6
- anaconda 4.5.10

#### 2 Windows

- python 3.6
- anaconda 4.5.10
- VS 2015


### 2 环境及服务安装

#### 2.1 Linux 环境

##### 1 安装 anaconda

- 版本要求：conda 4.5.10

> 安装过程：略（见anaconda官方安装手册）

> 验证: `conda -V`

```
root@lily:~/home/ai# conda -V
conda 4.5.10
root@lily:~/home/ai#
```

##### 2 安装python运行环境

```
# conda 创建一个python运行环境
conda create --name orc python=3.6

# 激活环境
source activate orc 

# 安装初始包
pip install numpy scipy matplotlib pillow -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install easydict opencv-python keras h5py PyYAML -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tensorflow -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install cython==0.24
pip install flask
```

##### 3 编译cython包

```
cd /ctpn/ctpnlib/utils
cython bbox.pyx
cython cython_nms.pyx
python setup_cpu.py build_ext --inplace
mv utils/* ./
rm -rf build
rm -rf utils
```

- linux setup_cpu.py
```
from Cython.Build import cythonize
import os
from os.path import join as pjoin
import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

def find_in_path(name, path):
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

def customize_compiler_for_nvcc(self):
    self.src_extensions.append('.cu')
    default_compiler_so = self.compiler_so
    super = self._compile
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        print(extra_postargs)
        postargs = extra_postargs['gcc']
        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so
    # inject our redefined _compile method into the class
    self._compile = _compile

# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)

ext_modules = [
    Extension(
        "utils.bbox",
        ["bbox.pyx"],
        extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
        include_dirs = [numpy_include]
    ),
    Extension(
        "utils.cython_nms",
        ["cython_nms.pyx"],
        extra_compile_args={'gcc': ["-Wno-cpp", "-Wno-unused-function"]},
        include_dirs = [numpy_include]
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={'build_ext': custom_build_ext},
)
```

##### 4 服务安装

- 服务包名称: `AI_OCR_V0.1.tar.gz`

```
step1:
mkdir $HOME/AI_OCR
cd $HOME/AI_OCR

step2:
将AI_OCR_V0.1.tar.gz move 至 $HOME/AI_OCR
解压 tar -zxvf AI_OCR_V0.1.tar.gz
```

##### 5 启动

```
cd $HOME/AI_OCR
sh start.sh
```

##### 6 绿灯测试

```
cd $HOME/AI_OCR
sh greenlight.sh
```

- 结果如下
```
{"file": "./test_images/demo.jpg", "errorcode": "0", "errormsg": "OK", "items": [{"itemcoord": [144.0, 5.4437174797058105, 816.003173828125, 5.515200614929199, 143.99684143066406, 35.14686584472656, 816.0, 35.21834945678711], "itemconf": 0.9747254252433777, "words": "1987\u5e74\u80a1\u5e02\u5d29\u76d8\u53ef\u80fd\u662f\u73b0\u4ee3\u4ece\u4f17\u6548\u5e94\u5f15\u8d77\u7684\u7b2c\u4e00\u573a\u5371\u673a\u3002\u91d1\u878d\u4e1a\u666e\u53ca\u4e86\u52a8"}, {"itemcoord": [112.0, 41.317413330078125, 816.0013427734375, 41.345584869384766, 111.99867248535156, 74.49046325683594, 816.0, 74.51863098144531], "itemconf": 0.9881898760795593, "words": "\u6001\u6295\u8d44\u7ec4\u5408\u4fdd\u9669\u65b9\u6cd5\u2014\u6d89\u53ca\u4fdd\u62a4\u6295\u8d44\u8005\u907f\u514d\u6295\u8d44\u7ec4\u5408\u4e8f\u635f\u7684\u65b9\u6cd5\u3002\u8bb8\u591a\u673a\u6784\u63d0"}, {"itemcoord": [112.0, 79.88433074951172, 800.113525390625, 82.43319702148438, 111.886474609375, 110.5326919555664, 800.0, 113.08155059814453], "itemconf": 0.9816058874130249, "words": "\u4f9b\u8fd9\u79cd\u5b89\u5168\u7684\u4ea4\u6613\u65b9\u5f0f:\u5728\u5e02\u573a\u5411\u4e0b\u65f6\uff0c\u5356\u7a7a\u5e02\u573a;\u5728\u5e02\u573a\u8d70\u9ad8\u65f6\uff0c\u505a\u591a\u5e02\u573a"}, {"itemcoord": [111.99175262451172, 117.24832153320312, 624.0, 117.11056518554688, 112.0, 147.89735412597656, 624.0082397460938, 147.7595977783203], "itemconf": 0.9813819527626038, "words": "\u5f53\u7136\uff0c\u4e0a\u8ff0\u65b9\u6cd5\u53ea\u6709\u5e02\u573a\u4e00\u5c0f\u90e8\u5206\u4eba\u91c7\u7528\u65f6\uff0c\u624d\u884c\u4e4b\u6709\u6548\u3002"}, {"itemcoord": [144.0, 153.52159118652344, 816.0316772460938, 154.17616271972656, 143.9683074951172, 186.0606689453125, 816.0, 186.71524047851562], "itemconf": 0.9827093482017517, "words": "!\u4f46\u662f\u5982\u679c\u7b56\u7565\u8ffd\u968f\u8005\u53d8\u5f97\u5f88\u591a\uff0c\u6324\u6ee1\u6574\u4e2a\u5e02\u573a\u7a7a\u95f4\uff0c\u5e02\u573a\u5c31\u4f1a\u53d8\u5f97\u4e0d\u7a33\u5b9a\u3002"}, {"itemcoord": [111.93840026855469, 193.3380889892578, 800.0, 191.8654327392578, 112.0, 222.11935424804688, 800.0615844726562, 220.64669799804688], "itemconf": 0.9747151136398315, "words": "\u5f53\u5e02\u573a\u884c\u60c5\u4e0d\u597d\u65f6\uff0c\u5f88\u591a\u4eba\u5c31\u4f1a\u5e73\u4ed3,\u4f7f\u5f97\u5e02\u4ef7\u8d8a\u6765\u8d8a\u4f4e\uff0c\u6709\u65f6\u5c31\u4f1a\u5f15\u8d77\u5d29\u76d8:"}, {"itemcoord": [112.0, 229.0714569091797, 816.0027465820312, 229.1340789794922, 111.99726104736328, 259.89630126953125, 816.0, 259.95892333984375], "itemconf": 0.990266740322113, "words": "1987\u5e74\uff0c\u6709\u8bb8\u591a\u7684\u8ddf\u98ce\u8005\uff0c\u5e02\u573a\u7a7a\u95f4\u62e5\u6324\u4e0d\u582a\uff0c\u5f88\u591a\u5206\u6790\u6a21\u578b\u5e76\u6ca1\u6709\u5145\u5206\u8003\u8651"}, {"itemcoord": [112.0, 267.3511962890625, 240.58660888671875, 269.8958435058594, 111.41338348388672, 296.9942626953125, 240.0, 299.5389099121094], "itemconf": 0.9800471067428589, "words": "\u8fd9\u79cd\u4ece\u4f17\u73b0\u8c61"}, {"itemcoord": [144.0, 306.1710510253906, 800.07958984375, 307.9681091308594, 143.92041015625, 335.2298889160156, 800.0, 337.0269470214844], "itemconf": 0.9815490245819092, "words": "!\u4e8b\u969411\u5e74\u4e4b\u540e\u76841998\u5e74,\u53e6\u4e00\u573a\u5927\u5371\u673a\u968f\u5373\u800c\u81f3\uff0c\u5c06\u4fc4\u7f57\u65af\u5e02\u573a\u5377\u4eba\u5176\u4e2d"}, {"itemcoord": [112.0, 342.4230651855469, 816.0109252929688, 342.6776123046875, 111.98910522460938, 372.5558776855469, 816.0, 372.8104248046875], "itemconf": 0.9834849834442139, "words": "\u5bfc\u81f4\u8457\u540d\u5bf9\u51b2\u57fa\u91d1\u2014\u7f8e\u56fd\u957f\u671f\u8d44\u672c\u7ba1\u7406\u516c\u53f8\u5012\u95ed\u3002\u57281994\u5e74\uff0c\u7f8e\u56fd\u957f\u671f\u8d44\u672c"}, {"itemcoord": [112.0, 379.4111022949219, 816.00146484375, 379.443603515625, 111.9985122680664, 411.61480712890625, 816.0, 411.6473388671875], "itemconf": 0.9883561730384827, "words": "\u7ba1\u7406\u516c\u53f8\u662f\u6700\u5927\u7684\u5bf9\u51b2\u57fa\u91d1\u4e4b\u4e00\u3002\u5b83\u7684\u7ecf\u7406\u628a\u4ece\u6240\u7f57\u95e8\u5144\u5f1f\u90a3\u91cc\u5b66\u5230\u7684\u6280\u672f\u624b\u6bb5"}, {"itemcoord": [112.0, 418.1766357421875, 816.0386962890625, 419.101318359375, 111.9613037109375, 447.6405944824219, 816.0, 448.5652770996094], "itemconf": 0.9601036310195923, "words": "\u548c\u5b9a\u91cf\u5206\u6790\u8fd0\u7528\u5230\u6781\u81f4\u3002\u5f7c\u65f6\uff0c\u4ed6\u4eec\u662f\u65b0\u7684\u91d1\u878d\u5e02\u573a\u4e3b\u5bb0\uff0c\u4eba\u4eba\u90fd\u60f3\u4ece\u5176\u60ca\u4eba\u4e1a"}, {"itemcoord": [112.0, 454.2679138183594, 240.0729217529297, 454.5650634765625, 111.92707824707031, 485.6978454589844, 240.0, 485.9949951171875], "itemconf": 0.9792687296867371, "words": "\u7ee9\u4e2d\u5206\u4e00\u676f\u7fb9"}, {"itemcoord": [143.96426391601562, 494.6623229980469, 816.0, 493.8321228027344, 144.0, 523.5880737304688, 816.0357055664062, 522.7578735351562], "itemconf": 0.9734989404678345, "words": "!\u5f88\u5feb\uff0c\u5176\u4ed6\u673a\u6784\uff08\u5305\u62ec\u9ad8\u76db\u3001\u6469\u6839\u58eb\u4e39\u5229\u3001\u96f7\u66fc\u5144\u5f1f\u53ca\u8bb8\u591a\u65b0\u5bf9\u51b2\u57fa\u91d1\u7684\u81ea"}, {"itemcoord": [112.0, 531.6578979492188, 800.0361938476562, 532.4815063476562, 111.96379089355469, 561.9073486328125, 800.0, 562.73095703125], "itemconf": 0.9946443438529968, "words": "\u8425\u4ea4\u6613\u90e8\u95e8\uff09\u5f00\u59cb\u62c6\u89e3\u957f\u671f\u8d44\u672c\u7ba1\u7406\u516c\u53f8\u7684\u6295\u8d44\u7b56\u7565\u2014\u6bcf\u4e00\u79cd\u90fd\u6d89\u53ca\u6760\u6746\u5316"}, {"itemcoord": [111.96808624267578, 566.8378295898438, 800.0, 566.1689453125, 112.0, 599.6618041992188, 800.0319213867188, 598.992919921875], "itemconf": 0.995968759059906, "words": "\u4f17\u4eba\u8702\u62e5\u8fdb\u4eba\u5229\u6da6\u4e30\u539a\u7684\u76f8\u5bf9\u4ef7\u503c\u503a\u5238\u5957\u5229\u6295\u8d44\u9886\u57df\u3002\u91cf\u5316\u7b56\u7565\u7684\u8ddf\u98ce\u8005\u5145\u65a5\u4e86"}, {"itemcoord": [111.92919921875, 606.0446166992188, 816.0, 604.4586181640625, 112.0, 637.4750366210938, 816.07080078125, 635.8890380859375], "itemconf": 0.9865671396255493, "words": "\u8fd9\u4e2a\u5e02\u573a\u7a7a\u95f4\u3002\u98ce\u9669\u6a21\u578b\u4e0d\u518d\u90a3\u4e48\u7cbe\u786e\uff0c\u56e0\u4e3a\u5b83\u4e0d\u80fd\u53cd\u6620\u8fd9\u79cd\u4ece\u4f17\u73b0\u8c61\u53ca\u5176\u6f5c\u5728"}, {"itemcoord": [112.0, 642.8131103515625, 800.0897216796875, 644.8404541015625, 111.91029357910156, 673.2599487304688, 800.0, 675.2872314453125], "itemconf": 0.9936532378196716, "words": "\u5f71\u54cd\u3002\u9ad8\u6760\u6746\u5806\u8d77\u7684\u5934\u5bf8\u610f\u5473\u7740\u4e00\u6709\u98ce\u5439\u8349\u52a8\uff0c\u5c31\u4f1a\u5728\u77ed\u65f6\u95f4\u5185\u6467\u6bc1\u4e00\u5bb6\u516c\u53f8\u3002"}, {"itemcoord": [143.947998046875, 679.5347900390625, 816.0, 678.42431640625, 144.0, 711.0028686523438, 816.052001953125, 709.8923950195312], "itemconf": 0.9785917401313782, "words": "{1998\u5e747\u6708\uff0c\u5927\u578b\u6295\u8d44\u673a\u6784\u6240\u7f57\u95e8\u5144\u5f1f\u516c\u53f8\u5f00\u59cb\u5bf9\u4ee5\u5f80\u7684\u8ddf\u98ce\u5934\u5bf8\u8fdb\u884c\u51cf"}, {"itemcoord": [111.91547393798828, 718.2447509765625, 800.0, 716.3286743164062, 112.0, 748.5978393554688, 800.0845336914062, 746.6817626953125], "itemconf": 0.984032392501831, "words": "\u4ed3\u30021998\u5e748\u6708\uff0c\u4fc4\u7f57\u65af\u653f\u5e9c\u51fa\u73b0\u503a\u5238\u8fdd\u7ea6\u3002\u5f53\u76f8\u5bf9\u4ef7\u503c\u57fa\u91d1\u593a\u8def\u9003\u547d\u4e4b\u65f6"}, {"itemcoord": [112.0, 755.1364135742188, 816.095947265625, 757.438232421875, 111.90403747558594, 784.4883422851562, 816.0, 786.7901611328125], "itemconf": 0.9650871157646179, "words": "{\u4e00\u573a\u201c\u5927\u5730\u9707\u201d\u968f\u4e4b\u800c\u6765\u3002\u957f\u671f\u8d44\u672c\u7ba1\u7406\u516c\u53f8\u6fd2\u4e34\u7834\u4ea7;\u8bb8\u591a\u4eba\u5bb3\u6015\u8fd9\u4f1a\u7834\u574f\u6574"}, {"itemcoord": [111.96642303466797, 791.1217041015625, 800.0, 790.40234375, 112.0, 823.2363891601562, 800.0335693359375, 822.5170288085938], "itemconf": 0.98801189661026, "words": "\u4e2a\u91d1\u878d\u7cfb\u7edf\uff0c\u5c31\u59822008\u5e74\u96f7\u66fc\u5144\u5f1f\u516c\u53f8\u90a3\u6837\u3002\u7f8e\u8054\u50a8\u4ecb\u4eba,\u79c1\u4e0b\u534f\u8c03\u89e3\u51b3\u65b9\u6848:"}, {"itemcoord": [112.0, 832.02587890625, 288.0439147949219, 832.3145751953125, 111.95610046386719, 858.7957153320312, 288.0, 859.0844116210938], "itemconf": 0.8891814947128296, "words": "\u529b\u56fe\u904f\u5236\u6df7\u4e71\u5c40\u9762\u3002"}, {"itemcoord": [144.0, 869.2247924804688, 816.0364379882812, 870.11181640625, 143.9635772705078, 896.8162841796875, 816.0, 897.7033081054688], "itemconf": 0.9828020930290222, "words": "2000\u5e74\uff0c\u7f51\u7edc\u80a1\u7684\u5e02\u76c8\u7387\u9ad8\u81f3\u8352\u8c2c\u4e4b\u4f4d\uff0c\u6295\u8d44\u8005\u8702\u62e5\uff0c\u6ce1\u6cab\u5267\u589e\u3002\u52302000"}]}
```


### 3 服务版本目录介绍

目录或文件 | 说明
---|---
test_result | 测试图片结果目录
densenet  | 基于densenet的图像识别模型相关程序 
test_images | 测试图片目录
ctpn | 基于ctpn文本定位模型相关程序
pub | web服务公共模板
ocr_flask.py | 略
ocr_model | ctpn模型存放文件
ocr_model.py | orc识别主程序
__pycache__ | 
start.sh  | web服务启动脚本
greenlight.sh | web服务绿灯脚本
stop.sh | web服务停止脚本
check.sh | web服务检测脚本
api_test.py | web服务测试脚本
logs |日志目录
ocr_flask_1.py | flask服务启动脚本





### 1 参考腾讯模式

> http://open.youtu.qq.com/legency/#/develop/api-ocr-general

### 2 端到端文本识别API

- 请求包体

参数名 | 类型 | 说明
---|---|---
TraceId | string | 交易跟踪号
FileList | list string | 识别图片路径数组
Image | String(Bytes) | 需要检测的图像base64编码 jpg png bmp(预留)


- 例子

```
{
    "FileList":"nas/1.jpg"
    "Image":"",
    "TraceId":"20180808010101-0000001"
}
```

- 返回

参数名 | 二级参数 | 类型 | 说明
---|---|---|---
TraceId || string | 交易跟踪号
ErrorCode || int | 返回状态值
ErrorMsg  ||	String | 返回错误消息
File || list string | 识别图片路径
Items || Array | 识别出的所有字段信息每个字段包括，item, itemcoord, words
-|ItemCoord|Object|字段在图像中的像素坐标，包括左上角坐标x,y，以及宽、高width, height
-|ItemConf|Float|字段在图像中的像素坐标置信度
-|Words|string|识别结果string


- 返回报文正案例

```
{
    "ErrorCode":"0",
    "ErrorMsg":"OK",
    "File":"./test_images/demo.jpg",
    "Items":[
        {
            "ItemConf":0.984032392501831,
            "ItemCoord":[
                111.91547393798828,
                718.2447509765625,
                800.0,
                716.3286743164062,
                112.0,
                748.5978393554688,
                800.0845336914062,
                746.6817626953125
            ],
            "Words":"仓。1998年8月，俄罗斯政府出现债券违约。当相对价值基金夺路逃命之时"
        },
        {
            "ItemConf":0.9650871157646179,
            "ItemCoord":[
                112.0,
                755.1364135742188,
                816.095947265625,
                757.438232421875,
                111.90403747558594,
                784.4883422851562,
                816.0,
                786.7901611328125
            ],
            "Words":"{一场“大地震”随之而来。长期资本管理公司濒临破产;许多人害怕这会破坏整"
        }
    ],
    "TraceId":"20180808010101-0000001"
}
```

- 返回报文负案例

```
{
    "ErrorCode":"1",
    "ErrorMsg":"[Errno 2] No such file or directory: './test_images/demo1.jpg'",
    "File":"./test_images/demo1.jpg",
    "Items":[],
    "TraceId":"20180808010101-0000001"
}
```
