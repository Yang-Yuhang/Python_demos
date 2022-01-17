#修改图片格式
import os
import string


pathName = "D:\\课件\\人工智能技术\\test\\"

print("change the type:\n")
li=os.listdir(pathName)
for filename in li:
    newname = filename
    newname = newname.split(".")
    if newname[-1]=="jpg":   #图片的原格式的后缀
        newname[-1]="png"   #图片新格式后缀
        newname = str.join(".",newname)  #这里要用str.join
        filename = pathName+filename
        newname = pathName+newname
        os.rename(filename,newname)
        print(newname,"updated successfully")
print()
print("finish the change of types!\n")
    
