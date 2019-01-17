#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : MVLSTM 
@File     : logger_config.py
@Time     : 19-1-16 下午7:44
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""

import logging
import time
import os

base_logger = logging.getLogger("tk_logger")
base_logger.setLevel(logging.DEBUG)

curdir = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')

now_date = time.strftime("%Y_%m_%d")
fh = logging.FileHandler(curdir + '/web_server/logs/%s' % (now_date+'_dl.log'))
# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
# 定义handler的输出格式formatter
formatter = logging.Formatter("%(asctime)s %(pathname)s %(filename)s %(funcName)s %(lineno)s "
                              "%(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
fh.setFormatter(formatter)
ch.setFormatter(formatter)

base_logger.addHandler(fh)
base_logger.addHandler(ch)
