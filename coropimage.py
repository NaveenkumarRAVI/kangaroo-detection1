import cv2
import numpy as np
# example of extracting bounding boxes from an annotation file
from xml.etree import ElementTree
import os

# function to extract bounding boxes from an annotation file
def extract_boxes(filename):
    # load and parse the file
    tree = ElementTree.parse(filename)
    # get the root of the document
    root = tree.getroot()
    # extract each bounding box
    boxes = list()
    for box in root.findall('.//bndbox'):
        xmin = int(box.find('xmin').text)
        ymin = int(box.find('ymin').text)
        xmax = int(box.find('xmax').text)
        ymax = int(box.find('ymax').text)
        coors = [xmin, ymin, xmax, ymax]
        boxes.append(coors)
    # extract image dimensions
    width = int(root.find('.//size/width').text)
    height = int(root.find('.//size/height').text)
    return boxes, width, height


#based on the dimension crop file with only kangaroo images
for count,filename in enumerate(os.listdir("E://downloads/OID_Kangaroo_Pascal/OID_Kangaroo_Pascal/images/")):
    print(filename)
    src = 'E://downloads/OID_Kangaroo_Pascal/OID_Kangaroo_Pascal/images/'+filename
    print(count)
    img = cv2.imread(src)
    xmlname = filename.split('.')
    src1 = 'E://downloads/OID_Kangaroo_Pascal/OID_Kangaroo_Pascal/pascal/'+ xmlname[0]+'.xml'
    boxes,w,h = extract_boxes(src1)
    for x in range(len(boxes)):
        cropm = img[int(boxes[x][1]):int(boxes[x][3]),int(boxes[x][0]):int(boxes[x][2])]
        cv2.imwrite('E://downloads/OID_Kangaroo_Pascal/OID_Kangaroo_Pascal/croped/'+str(xmlname[0])+'__'+str(x)+'.jpg',cropm)

cv2.waitKey(0)
cv2.destroyAllWindows()

