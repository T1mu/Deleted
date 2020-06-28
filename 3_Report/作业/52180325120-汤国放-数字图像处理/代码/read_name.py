import os

def read_name(f_path,t_path):
            # 替换为你的路径
    dir = os.listdir(f_path)                  # dir是目录下的全部文件
    fopen = open(t_path, 'w') # 替换为你的路径
    for d in dir:                        # d是每一个文件的文件名
        string =  os.path.splitext(d)[0] + '\n'    #拼接字符串并换行
        fopen.write(string)             # 写入文件中
    fopen.close()

f_path = 'E://2019-6-26-YOLO//yolov3//VOCdevkit//VOC2018//Annotations' 
t_path = 'E://2019-6-26-YOLO//yolov3//VOCdevkit//VOC2018//ImageSets//Main//train.txt'
read_name(f_path,t_path)
t_path = 'E://2019-6-26-YOLO//yolov3//VOCdevkit//VOC2018//ImageSets//Main//val.txt'
read_name(f_path,t_path)
t_path = 'E://2019-6-26-YOLO//yolov3//VOCdevkit//VOC2018//ImageSets//Main//test.txt'
read_name(f_path,t_path)