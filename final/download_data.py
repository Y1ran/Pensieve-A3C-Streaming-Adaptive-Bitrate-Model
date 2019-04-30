# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 13:13:32 2018

@author: Administrator
"""

import wget, time
import os
 
# 网络地址

DATA_URL = 'http://164.52.0.183:8000/file/findTrace/2018-12-24.txt'
# DATA_URL = '/home/xxx/book/data.tar.gz'
out_fname = '2018-12-24.txt'

def download(DATA_URL):
    out_fname = '2018-12-24.txt'
    date = time.ctime()
    path = str(date.split(' ')[1] +'-' + date.split(' ')[2])
    wget.download(DATA_URL, out=out_fname)
    
    if not os.path.exists('./' + out_fname):
        wget.download(DATA_URL, out=out_fname)
    else:
        #os.remove('./' + out_fname)
        print("today's data has been download")
    mkdir(path)
    
    return path

def mkdir(path):
    # 去除首位空格
 
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists=os.path.exists(path)
 
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        os.makedirs(path) 
 
        print(path +' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path +' 目录已存在')
        return False
# 提取压缩包
#tar = tarfile.open(out_fname)
#tar.extractall()
#tar.close()
# 删除下载文件
#os.remove(out_fname)
        
# 调用函数
path = download(DATA_URL)


file = open("./" + out_fname)
lines = file.readlines()
output = {}
temp = ""
cnt = 0
for line in lines:
    line=line.strip('\n')
    if line.startswith("FPS"): 
        fps_split = line.split("=")
        #print(fps_split)
        fps_temp = fps_split[1]
        for i in range(1,cnt+1):
            output[temp][-i] += " "+fps_temp
        cnt = 0
    elif line.startswith("ID:dokidoki/mlinkm/"):
        Channel_ID_1200 = line[19:] 
        if Channel_ID_1200 in output:
            temp = Channel_ID_1200 + "_high" 
        else:
            output[Channel_ID_1200 + "_high"] = []
            temp = Channel_ID_1200 + "_high"
        cnt = 0
    elif line.startswith("ID:EXT-ENC-0/dokidoki/mlinkm/"):
        Channel_ID_500 = line[29:]
        if Channel_ID_1200 in output:
            temp = Channel_ID_500 + "_low" 
        else:
            output[Channel_ID_500 + "_low"] = []
            temp = Channel_ID_500 + "_low"
        cnt = 0
    else:
        output[temp].append(line) 
        cnt += 1
for key,value in output.items():
    f_file = open("./" + path + "/" + str(key) + ".csv","w")
    for idx in range(len(value)):
        data = value[idx].replace(" ",",")
        data += "\n"
        f_file.write(data)
#print(output)
        #print(Channel_ID_500)
