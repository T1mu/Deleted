from PIL import Image
import os.path
import glob
def convertjpg(jpgfile,outdir,width=608,height=608):
    img=Image.open(jpgfile)
    try:
        new_img=img.resize((width,height),Image.BILINEAR) 
        new_img = new_img.rotate(270)  
        new_img.save(os.path.join(outdir,os.path.basename(jpgfile)))
    except Exception as e:
        print(e)
for jpgfile in glob.glob("E:\\square-p\\2019_07\\*.jpg"):
    convertjpg(jpgfile,"E:\\2019-6-26-YOLO\\yolov3\\VOCdevkit\\VOC2018\\JPEGImages")