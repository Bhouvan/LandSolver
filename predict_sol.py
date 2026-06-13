from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import sys

# Charger le modèle avec gestion d'erreur
try:
    model = load_model('soil_model.keras')  # Version optimisée d'abord
except:
    model = load_model('soil_model.h5')     # Fallback

classes = ['Alluvial', 'Arid', 'Black', 'Laterite', 'Mountain', 'Red', 'Yellow']

# Permettre de spécifier une image en argument
if len(sys.argv) > 1:
    img_path = sys.argv[1]
else:
    img_path = 'imagetest/solargileux-1.jpg'

try:
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    classe = classes[np.argmax(prediction)]
    confiance = np.max(prediction) * 100
    
    print(f"📷 Image : {img_path}")
    print(f"🌱 Type de sol : {classe}")
    print(f"🎯 Confiance : {confiance:.1f}%")
    
    # Afficher les probabilités détaillées
    print("\n📊 Probabilités détaillées :")
    for i, c in enumerate(classes):
        print(f"  {c:10s} : {prediction[0][i]*100:.1f}%")
        
except Exception as e:
    print(f"❌ Erreur : {e}")
    print(f"Usage: python predict_sol.py [chemin_image]")