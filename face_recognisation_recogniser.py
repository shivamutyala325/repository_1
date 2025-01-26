import cv2
import os


persons = []
today = []

faces_dir = r"path_of_your_folder_containg_images"
for i in os.listdir(faces_dir):
    persons.append(i)

classifier = cv2.CascadeClassifier(r"C:\Users\shiva\PycharmProjects\Image_processing\venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trained_data1.yml')


video = cv2.VideoCapture(0)

try:
    while True:
        res, frame = video.read()
        if not res:
            print("Failed to capture video frame.")
            break


        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        faces = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)

        for (x, y, w, h) in faces:

            face_roi = gray[y:y + h, x:x + w]

            try:

                tag, confidence = recognizer.predict(face_roi)
            except cv2.error as e:
                print(f"Error during prediction: {e}")
                continue


            if confidence < 50:
                if persons[tag] not in today:
                    print(f"{persons[tag]} is present")
                    today.append(persons[tag])


            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


        cv2.imshow("Student Attendance", frame)

        if cv2.waitKey(20) & 0xFF == ord('s'):
            break


    for person in persons:
        if person not in today:
            print(f"{person} is absent")

finally:
    video.release()
    cv2.destroyAllWindows()


