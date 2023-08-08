import cv2, os

cascPath = r"C:\Users\Administrator\Desktop\FaceDetection\haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

live = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = live.read()

    # Converting frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # using haarcascade
    faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(1,1),
    flags = cv2.CASCADE_SCALE_IMAGE)

    os.system("cls")
    print("Detected {0} faces!".format(len(faces)))


    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Faces Detected", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
live.release()
cv2.destroyAllWindows()