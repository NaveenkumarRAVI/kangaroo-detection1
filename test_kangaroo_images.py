# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# construct the argument parser and parse the arguments
# load the image
image = cv2.imread('/test image/image3.jpg')

orig = image.copy()
# pre-process the image for classification
image = cv2.resize(image, (224, 224))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
# load the trained convolutional neural network
print("[INFO] loading network...")
kangaru_model = load_model('kangaroo.model')
# classify the input image
(not_kangaroo,kangaroo) = kangaru_model.predict(image)[0]
print(not_kangaroo,kangaroo)
# build the label
label = "kangaroo" if (kangaroo > not_kangaroo) else "Not kangaroo"
proba = kangaroo if (kangaroo > not_kangaroo) else not_kangaroo
label = "{}: {:.2f}%".format(label, proba * 100)
# draw the label on the image
print(label)
cv2.putText(orig, label, (20, 30),  cv2.FONT_HERSHEY_SIMPLEX,.7, (0, 0,255), 2)

plt.axis("off")
plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
plt.show()
# show the output image
