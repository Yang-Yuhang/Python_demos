import os
from PIL import  Image

path = "D:\\课件\\人工智能技术\\test\\"   #操作路径
filelist = os.listdir(path)  #获取路径文件列表

#修改图片名字
count=1
print("change the file name:\n")
for file in filelist:
    print("source file:",file)    #显示原图片名称和后缀
for file in filelist:
    Olddir=os.path.join(path,file)  #读取每张源图片地址、名称、类型
    if os.path.isdir(Olddir):
        continue
    filename=os.path.splitext(file)[0]   #文件名，勿更改
    filetype=os.path.splitext(file)[1]   #文件类型，勿更改
    Newdir=os.path.join(path,'1'+str(count).zfill(4)+filetype)  #新名称
    os.rename(Olddir,Newdir)   #重命名
    print("update name successfully:",Newdir)
    count+=1
print()
print("successfully change all file names!\n")


#修改图片大小
print("change the size:\n")
for file in filelist:
    print("source file:",file)
for maindir,subdir,file_name_list in os.walk(path):
    for file_name in file_name_list:
        image=os.path.join(maindir,file_name) #获取每张图片的路径
        file=Image.open(image)
        out=file.resize((128,128),Image.ANTIALIAS)  #以高质量修改图片尺寸为（128,128）
        print("update size successfully:",maindir,file_name)
        out.save(image)   #以同名保存到原路径
print()
print("finish the change of sizes!\n")
    










