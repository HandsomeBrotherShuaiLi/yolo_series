import xml.etree.ElementTree as ET
import os
classes=['point','line']
path="data/annotation/"
def convert_annotation(path):
    xml_list=os.listdir(path)
    traintxt=open('data/train.txt',encoding='utf-8',mode='w')
    for xml_file in xml_list:
        tree=ET.parse(path+xml_file)
        root=tree.getroot()
        filename=root.find('filename').text
        traintxt.write('data/image/'+filename)
        for obj in root.iter('object'):
            difficult=obj.find('difficult').text
            cls=obj.find('name').text
            if cls not in classes or int(difficult)==1:
                continue
            cls_id=classes.index(cls)
            xmlbox=obj.find('bndbox')
            b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
                 int(xmlbox.find('ymax').text))
            traintxt.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        traintxt.write('\n')
if __name__=="__main__":
    convert_annotation(path)

