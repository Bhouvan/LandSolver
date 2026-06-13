from extract_irregularites import extraire_irregularites
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import joblib
import threading

try:
    model = load_model('soil_model.keras')
except:
    try:
        model = load_model('soil_model.h5')
    except:
        model = None
        print("ERREUR: Modèle non trouvé.")

try:
    scaler = joblib.load('scaler.pkl')  # ✅ chargement du scaler
except:
    scaler = None
    print("ERREUR: scaler.pkl non trouvé. Relancez la préparation du dataset.")

classes = ['Alluvial', 'Arid', 'Black', 'Laterite', 'Mountain', 'Red', 'Yellow']

class SoilApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Classification des Sols")
        self.root.geometry("650x600")
        self.root.resizable(False, False)

        if model is None or scaler is None:
            tk.Label(root,
                     text="⚠️ Modèle ou scaler non trouvé !\nLancez d'abord la préparation et l'entraînement.",
                     fg="red", font=("Arial", 12)).pack(pady=50)
            return

        self.label_titre = tk.Label(root, text="🌱 Classification des Sols 🌱",
                                    font=("Arial", 20, "bold"), fg="green")
        self.label_titre.pack(pady=10)

        self.btn_charger = tk.Button(root, text="📂 Charger une image",
                                     command=self.charger_image,
                                     font=("Arial", 12), bg="lightblue", width=20)
        self.btn_charger.pack(pady=10)

        self.label_image = tk.Label(root, relief="solid", borderwidth=2)
        self.label_image.pack(pady=10)

        self.btn_predire = tk.Button(root, text="🔍 Prédire le type de sol",
                                     command=self.predire,
                                     font=("Arial", 12), bg="orange",
                                     state=tk.DISABLED, width=25)
        self.btn_predire.pack(pady=10)

        self.label_resultat = tk.Label(root, text="", font=("Arial", 12),
                                       justify="left", fg="blue")
        self.label_resultat.pack(pady=10)

        self.chemin_image = None

    def charger_image(self):
        chemin = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
        )
        if chemin:
            self.chemin_image = chemin
            img = Image.open(chemin)
            img.thumbnail((300, 300))
            self.photo = ImageTk.PhotoImage(img)
            self.label_image.config(image=self.photo)
            self.btn_predire.config(state=tk.NORMAL)
            self.label_resultat.config(text="")
            self.root.title(f"Classification des Sols - {chemin.split('/')[-1]}")

    def predire(self):
        if not self.chemin_image:
            return
        self.btn_predire.config(state=tk.DISABLED, text="⏳ Prédiction en cours...")
        self.root.update()
        threading.Thread(target=self._faire_prediction).start()

    def _faire_prediction(self):
        try:
            img = image.load_img(self.chemin_image, target_size=(128, 128))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            irr = extraire_irregularites(self.chemin_image)
            if irr is None:
                raise Exception("Impossible d'extraire les irrégularités")

            irr_array = np.array(irr).reshape(1, -1)
            irr_array = scaler.transform(irr_array)  # ✅ normalisation

            prediction = model.predict([img_array, irr_array])
            idx = np.argmax(prediction)
            classe = classes[idx]
            confiance = prediction[0][idx] * 100

            pourcentages = ""
            for i, nom in enumerate(classes):
                proba = prediction[0][i] * 100
                barre = "█" * int(proba / 10) + "░" * (10 - int(proba / 10))
                pourcentages += f"{nom:10s} : {barre} {proba:.1f}%\n"

            self.root.after(0, self._mettre_a_jour_resultat, classe, confiance, pourcentages)

        except Exception as e:
            self.root.after(0, self._afficher_erreur, str(e))

    def _mettre_a_jour_resultat(self, classe, confiance, pourcentages):
        self.label_resultat.config(
            text=f"🏆 Résultat : {classe} (confiance : {confiance:.1f}%)\n\n"
                 f"📊 Détail des probabilités :\n{pourcentages}"
        )
        self.btn_predire.config(state=tk.NORMAL, text="🔍 Prédire le type de sol")
        self.root.title(f"Classification des Sols - Résultat : {classe}")

    def _afficher_erreur(self, erreur):
        messagebox.showerror("Erreur", f"Impossible de traiter l'image :\n{erreur}")
        self.btn_predire.config(state=tk.NORMAL, text="🔍 Prédire le type de sol")

if __name__ == "__main__":
    root = tk.Tk()
    app = SoilApp(root)
    root.mainloop()