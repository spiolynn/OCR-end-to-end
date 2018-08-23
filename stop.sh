ps -ef | grep ocr_flask_1.py | grep python | grep -v grep | awk '{print $2}' |xargs kill -9 
