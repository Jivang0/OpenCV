import cv2
# Load Haar Cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
# Start webcam
cap = cv2.VideoCapture(0)
save_count = 0   # counter for saved frames
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        # Detect eyes in face ROI
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)
    # Show result
    cv2.imshow("Face & Eye Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    # Press 's' to save the current frame
    if key == ord('s'):
        save_name = f"saved_frame_{save_count}.jpg"
        cv2.imwrite(save_name, frame)
        print(f"Saved: {save_name}")
        save_count += 1
    # Press 'q' to quit
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
