import cv2
import numpy as np
import os

# Yüz resimlerinin bulunduğu klasör
known_faces_dir = r"C:\Users\TozLu\Desktop\furkandogan"

# Yüz tanıma için kullanılan Haarcascade dosyasını yükleyin
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Bilinen yüzleri ve etiketlerini saklama
known_face_encodings = []
known_face_names = []

# Klasördeki yüz resimlerini işleyin
for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        filepath = os.path.join(known_faces_dir, filename)
        image = cv2.imread(filepath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Yüzü algıla
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_encoding = gray[y:y + h, x:x + w]  # Yüz bölgesini al
            known_face_encodings.append(face_encoding)
            known_face_names.append(os.path.splitext(filename)[0])  # Dosya adını isim olarak kullan

# Kamerayı başlat
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_to_check = gray[y:y + h, x:x + w]  # Algılanan yüz

        # Yüzü karşılaştırma
        name = "Bilinmeyen"
        for i, known_face in enumerate(known_face_encodings):
            diff = cv2.absdiff(cv2.resize(known_face, (w, h)), cv2.resize(face_to_check, (w, h)))
            similarity = np.sum(diff) / (w * h)
            if similarity < 50:  # Eşik değeri ayarlanabilir
                name = known_face_names[i]
                break

        # Yüzü çerçevele ve etiket ekle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (191, 255, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 191, 0), 2)

    cv2.imshow("Yüz Algılama ve Tanıma", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
