from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('soil_model.h5')

classes = ['Alluvial', 'Arid', 'Black', 'Laterite', 'Mountain', 'Red', 'Yellow']

img_path = 'imagetest/solargileux-1.jpg'
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
classe = classes[np.argmax(prediction)]
print(f"Type de sol : {classe}")