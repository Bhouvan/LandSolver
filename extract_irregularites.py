import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import StandardScaler
import pickle
import joblib
from skimage.feature import graycomatrix, graycoprops  # import en haut, pas dans la boucle


def extraire_irregularites(img_path):
    img = cv2.imread(img_path, 0)
    if img is None:
        return None

    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0).var()
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1).var()

    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    entropie = -np.sum(hist * np.log2(hist + 1e-7))

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    rugosite = np.std(img.astype(np.float32) - blur.astype(np.float32))

    glcm = graycomatrix(img, [1], [0], 256, symmetric=True)
    contraste = graycoprops(glcm, 'contrast')[0, 0]
    homogeneite = graycoprops(glcm, 'homogeneity')[0, 0]
    energie = graycoprops(glcm, 'energy')[0, 0]

    return [laplacian_var, grad_x, grad_y, entropie, rugosite, contraste, homogeneite, energie]


def preparer_dataset_avec_irregularites(data_dir):
    classes = sorted([
        c for c in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, c))
    ])

    X_images = []
    X_irregularites = []
    y = []
    skipped = 0

    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        print(f"Traitement de {class_name}...")

        for img_file in os.listdir(class_dir):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(class_dir, img_file)

            irr = extraire_irregularites(img_path)
            if irr is None:  # ✅ on vérifie AVANT de charger l'image
                skipped += 1
                continue

            img_loaded = image.load_img(img_path, target_size=(128, 128))
            img_array = image.img_to_array(img_loaded) / 255.0

            X_images.append(img_array)
            X_irregularites.append(irr)
            y.append(class_idx)

    print(f"{skipped} images ignorées")

    X_irr_array = np.array(X_irregularites)

    # Normalisation
    scaler = StandardScaler()
    X_irr_normalized = scaler.fit_transform(X_irr_array)
    joblib.dump(scaler, 'scaler.pkl')  # nécessaire pour la prédiction

    return (np.array(X_images), X_irr_normalized, np.array(y), classes)


if __name__ == "__main__":
    print("Préparation du dataset...")
    X_img, X_irr, y, classes = preparer_dataset_avec_irregularites('Soil-Classification-Dataset')

    with open('dataset_irregularites.pkl', 'wb') as f:
        pickle.dump((X_img, X_irr, y, classes), f)

    print(f"Dataset sauvegardé !")
    print(f"Images: {X_img.shape}")
    print(f"Irrégularités: {X_irr.shape}")
    print(f"Classes: {classes}")