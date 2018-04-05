import os
import xml.etree.ElementTree as ET
import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt

# %config InlineBackend.figure_format = 'svg'

options = {
    'model': 'cfg/yolo_network_train.cfg',
    'load': 'bin/yolo_network_train_2000.weights',
    'threshold': 0.2
}

tfnet = TFNet(options)

# read the color image and covert to RGB

img = cv2.imread('./images/000090.jpg', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = tfnet.return_predict(img)
print(results)


tree = ET.parse('./annotations/000090.xml')
root = tree.getroot()


bnd_box = tree.findall('object/bndbox')
loob = []
for j in (bnd_box):
    t1 = int(j.find('xmin').text)
    t2 = int(j.find('ymin').text)
    b1 = int(j.find('xmax').text)
    b2 = int(j.find('ymax').text)
    tl1 = (t1, t2)
    br1 = (b1, b2)
    img = cv2.rectangle(img,tl1,br1,(255,255,0), 5)
    img = cv2.putText(img, "Training", tl1, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0),1)
    loob.append(img)

for i in range(0,len(results)):
    tl = (results[i]['topleft']['x'], results[i]['topleft']['y'])
    br = (results[i]['bottomright']['x'], results[i]['bottomright']['y'])
    label = results[i]['label']
    accuracy = results[i]['confidence']
    acc = str("%.02f" % accuracy)
    img=cv2.rectangle(img, tl, br,(0, 255 ,0), 5)
    img=cv2.putText(img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
    img=cv2.putText(img,acc, br,cv2.FONT_HERSHEY_COMPLEX, 1.5,(255,0,0),2)
    loob.append(img)
plt.imshow(img)
plt.show()
