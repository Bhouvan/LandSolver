import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Charger le modèle
model = load_model('soil_model.h5')
classes = ['Alluvial', 'Arid', 'Black', 'Laterite', 'Mountain', 'Red', 'Yellow']

class SoilApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Classification des Sols")
        self.root.geometry("600x500")
        
        # Widgets
        self.label_titre = tk.Label(root, text="Classification des Sols", font=("Arial", 20))
        self.label_titre.pack(pady=10)
        
        self.btn_charger = tk.Button(root, text="Charger une image", command=self.charger_image, font=("Arial", 12))
        self.btn_charger.pack(pady=10)
        
        self.label_image = tk.Label(root)
        self.label_image.pack(pady=10)
        
        self.btn_predire = tk.Button(root, text="Prédire le type de sol", command=self.predire, font=("Arial", 12), state=tk.DISABLED)
        self.btn_predire.pack(pady=10)
        
        self.label_resultat = tk.Label(root, text="", font=("Arial", 14, "bold"))
        self.label_resultat.pack(pady=10)
        
        self.chemin_image = None
        
    def charger_image(self):
        chemin = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if chemin:
            self.chemin_image = chemin
            img = Image.open(chemin)
            img = img.resize((250, 250))
            self.photo = ImageTk.PhotoImage(img)
            self.label_image.config(image=self.photo)
            self.btn_predire.config(state=tk.NORMAL)
            self.label_resultat.config(text="")
    
    def predire(self):
        if not self.chemin_image:
            return
        
        img = image.load_img(self.chemin_image, target_size=(128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array)
        classe = classes[np.argmax(prediction)]
        pourcentages = "\n".join([f"{classes[i]}: {prediction[0][i]*100:.1f}%" for i in range(7)])
        
        self.label_resultat.config(text=f"Résultat : {classe}\n\nProbabilités :\n{pourcentages}")

# Lancer l'application
if __name__ == "__main__":
    root = tk.Tk()
    app = SoilApp(root)
    root.mainloop()