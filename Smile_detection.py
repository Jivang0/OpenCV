import cv2 as cv

face_cascade = cv.CascadeClassifier(cv.data.haarcascades +'haarcascade_frontalface_default.xml')


# smile_cascade = cv.CascadeClassifier(cv.data.haarcascades+ "haarcascade_smile.xml")
smile_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_smile.xml")

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]

        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))

        for (sx, sy, sw, sh) in smiles:
            cv.rectangle(frame, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (0, 255, 0), 2)

    cv.imshow('Smile Detection', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
