from tkinter import *
from PIL import Image, ImageTk
from tkinter import messagebox
import face_recognition
import cv2
import mysql.connector
from datetime import datetime
import threading
import os


class FaceRecognition:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1530x790+0+0")
        self.root.title("Face Recognition System")

        # Title
        title_lbl = Label(self.root, text="FACE RECOGNITION", font=("times new roman", 35, "bold"), bg="white", fg="green")
        title_lbl.place(x=0, y=0, width=1530, height=45)

        # Top Image
        img_top = Image.open("collage_images/fac.jpg")
        img_top = img_top.resize((650, 700), Image.ANTIALIAS)
        self.photoimg_top = ImageTk.PhotoImage(img_top)

        f_lbl_top = Label(self.root, image=self.photoimg_top)
        f_lbl_top.place(x=0, y=55, width=650, height=700)

        # Bottom Image
        img_bottom = Image.open("collage_images/fac4.jpg")
        img_bottom = img_bottom.resize((950, 700), Image.ANTIALIAS)
        self.photoimg_bottom = ImageTk.PhotoImage(img_bottom)

        f_lbl_bottom = Label(self.root, image=self.photoimg_bottom)
        f_lbl_bottom.place(x=650, y=55, width=950, height=700)

        # Button
        b1_1 = Button(f_lbl_bottom, text="Face Recognition", cursor="hand2", font=("times new roman", 18, "bold"),
                      bg="darkgreen", fg="white", command=self.run_face_recognition)
        b1_1.place(x=365, y=620, width=200, height=40)

    def mark_attendance(self, id, roll, name, department):
        try:
            with open("attendance.csv", "r+", newline="\n") as f:
                attendance_data = f.readlines()
                recorded_ids = {line.split(",")[0] for line in attendance_data}

                if str(id) not in recorded_ids:
                    now = datetime.now()
                    date, time = now.strftime("%d/%m/%Y"), now.strftime("%H:%M:%S")
                    f.write(f"{id},{roll},{name},{department},{time},{date},Present\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to mark attendance: {e}", parent=self.root)

    def run_face_recognition(self):
        threading.Thread(target=self.face_recog).start()

    def face_recog(self):
        def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, clf):
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors, minSize=(50, 50))

            for (x, y, w, h) in features:
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)

                # Recognize face
                face = gray_image[y:y + h, x:x + w]
                face = cv2.resize(face, (200, 200))
                id, predict = clf.predict(face)
                confidence = int(100 * (1 - predict / 300))  # Confidence calculation

                # Fetch details from the database
                try:
                    conn = mysql.connector.connect(
                        host="localhost", username="root", password="test123", database="face_recognizer"
                    )
                    my_cursor = conn.cursor()

                    my_cursor.execute(f"SELECT Name FROM student WHERE Student_id={id}")
                    name_result = my_cursor.fetchone()
                    n = name_result[0] if name_result else "Unknown Student"

                    my_cursor.execute(f"SELECT Roll FROM student WHERE Student_id={id}")
                    roll_result = my_cursor.fetchone()
                    r = roll_result[0] if roll_result else "Unknown Roll"

                    my_cursor.execute(f"SELECT Dep FROM student WHERE Student_id={id}")
                    dep_result = my_cursor.fetchone()
                    d = dep_result[0] if dep_result else "Unknown Department"

                    conn.close()

                    if confidence > 60:  # Recognition threshold
                        cv2.putText(img, f"ID: {id}", (x, y - 75), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
                        cv2.putText(img, f"Roll: {r}", (x, y - 55), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
                        cv2.putText(img, f"Name: {n}", (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
                        cv2.putText(img, f"Department: {d}", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
                        self.mark_attendance(id, r, n, d)
                    else:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
                        cv2.putText(img, "Unknown Face", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)

                except mysql.connector.Error as db_err:
                    print(f"Database Error: {db_err}")

        def recognize(img, clf, faceCascade):
            draw_boundary(img, faceCascade, 1.1, 10, (255, 0, 255), clf)
            return img

        if not os.path.exists("classifier.xml"):
            messagebox.showerror("Error", "Classifier file not found! Train the model first.", parent=self.root)
            return

        # Load Haar Cascade and LBPH Classifier
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
        try:
            clf = cv2.face.LBPHFaceRecognizer_create()  # Correct way to initialize LBPH
            clf.read("classifier.xml")
        except AttributeError:
            messagebox.showerror("Error", "LBPH Face Recognizer not found. Please install opencv-contrib-python", parent=self.root)
            return

        video_cap = cv2.VideoCapture(0)

        try:
            while True:
                ret, img = video_cap.read()
                if not ret:
                    break
                img = recognize(img, clf, faceCascade)
                cv2.imshow("Face Recognition", img)

                if cv2.waitKey(1) == 13:  # Press 'Enter' to exit
                    break
        finally:
            video_cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    root = Tk()
    obj = FaceRecognition(root)
    root.mainloop()


#####
#HACK CODE
#####


# from tkinter import *
# from PIL import Image, ImageTk
# from tkinter import messagebox
# import face_recognition
# import cv2
# import mysql.connector
# from datetime import datetime
# import threading
# import os


# class FaceRecognition:
#     def __init__(self, root):
#         self.root = root
#         self.root.geometry("1530x790+0+0")
#         self.root.title("Face Recognition System")

#         # Title
#         title_lbl = Label(self.root, text="FACE RECOGNITION", font=("times new roman", 35, "bold"), bg="white", fg="green")
#         title_lbl.place(x=0, y=0, width=1530, height=45)

#         # Top Image
#         img_top = Image.open("collage_images/fac.jpg")
#         img_top = img_top.resize((650, 700), Image.ANTIALIAS)
#         self.photoimg_top = ImageTk.PhotoImage(img_top)

#         f_lbl_top = Label(self.root, image=self.photoimg_top)
#         f_lbl_top.place(x=0, y=55, width=650, height=700)

#         # Bottom Image
#         img_bottom = Image.open("collage_images/fac4.jpg")
#         img_bottom = img_bottom.resize((950, 700), Image.ANTIALIAS)
#         self.photoimg_bottom = ImageTk.PhotoImage(img_bottom)

#         f_lbl_bottom = Label(self.root, image=self.photoimg_bottom)
#         f_lbl_bottom.place(x=650, y=55, width=950, height=700)

#         # Button
#         b1_1 = Button(f_lbl_bottom, text="Face Recognition", cursor="hand2", font=("times new roman", 18, "bold"),
#                       bg="darkgreen", fg="white", command=self.run_face_recognition)
#         b1_1.place(x=365, y=620, width=200, height=40)

#     def mark_attendance(self, id, roll, name, department):
#         try:
#             with open("attendance.csv", "r+", newline="\n") as f:
#                 attendance_data = f.readlines()
#                 recorded_ids = {line.split(",")[0] for line in attendance_data}

#                 if str(id) not in recorded_ids:
#                     now = datetime.now()
#                     date, time = now.strftime("%d/%m/%Y"), now.strftime("%H:%M:%S")
#                     f.write(f"{id},{roll},{name},{department},{time},{date},Present\n")
#         except Exception as e:
#             messagebox.showerror("Error", f"Failed to mark attendance: {e}", parent=self.root)

#     def run_face_recognition(self):
#         threading.Thread(target=self.face_recog).start()

#     def face_recog(self):
#         def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, clf):
#             gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors, minSize=(50, 50))

#             for (x, y, w, h) in features:
#                 cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)

#                 # Recognize face
#                 face = gray_image[y:y + h, x:x + w]
#                 face = cv2.resize(face, (200, 200))
#                 id, predict = clf.predict(face)
#                 confidence = int(100 * (1 - predict / 300))  # Confidence calculation

#                 # Fetch details from the database
#                 try:
#                     conn = mysql.connector.connect(
#                         host="localhost", username="root", password="test123", database="face_recognizer"
#                     )
#                     my_cursor = conn.cursor()

#                     my_cursor.execute(f"SELECT Name FROM student WHERE Student_id={61}")
#                     name_result = my_cursor.fetchone()
#                     n = name_result[0] if name_result else "Unknown Student"

#                     my_cursor.execute(f"SELECT Roll FROM student WHERE Student_id={61}")
#                     roll_result = my_cursor.fetchone()
#                     r = roll_result[0] if roll_result else "Unknown Roll"

#                     my_cursor.execute(f"SELECT Dep FROM student WHERE Student_id={61}")
#                     dep_result = my_cursor.fetchone()
#                     d = dep_result[0] if dep_result else "Unknown Department"

#                     conn.close()

#                     if confidence > 60:  # Recognition threshold
#                         cv2.putText(img, f"ID: {id}", (x, y - 75), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
#                         cv2.putText(img, f"Roll: {r}", (x, y - 55), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
#                         cv2.putText(img, f"Name: {n}", (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
#                         cv2.putText(img, f"Department: {d}", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
#                         self.mark_attendance(id, r, n, d)
#                     else:
#                         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
#                         cv2.putText(img, "Unknown Face", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)

#                 except mysql.connector.Error as db_err:
#                     print(f"Database Error: {db_err}")

#         def recognize(img, clf, faceCascade):
#             draw_boundary(img, faceCascade, 1.1, 10, (255, 0, 255), clf)
#             return img

#         if not os.path.exists("classifier.xml"):
#             messagebox.showerror("Error", "Classifier file not found! Train the model first.", parent=self.root)
#             return

#         # Load Haar Cascade and LBPH Classifier
#         faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
#         try:
#             clf = cv2.face.LBPHFaceRecognizer_create()  # Correct way to initialize LBPH
#             clf.read("classifier.xml")
#         except AttributeError:
#             messagebox.showerror("Error", "LBPH Face Recognizer not found. Please install opencv-contrib-python", parent=self.root)
#             return

#         video_cap = cv2.VideoCapture(0)

#         try:
#             while True:
#                 ret, img = video_cap.read()
#                 if not ret:
#                     break
#                 img = recognize(img, clf, faceCascade)
#                 cv2.imshow("Face Recognition", img)

#                 if cv2.waitKey(1) == 13:  # Press 'Enter' to exit
#                     break
#         finally:
#             video_cap.release()
#             cv2.destroyAllWindows()


# if __name__ == "__main__":
#     root = Tk()
#     obj = FaceRecognition(root)
#     root.mainloop()


