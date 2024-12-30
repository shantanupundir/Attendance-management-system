from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import os
import numpy as np


class Train:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1530x790+0+0")
        self.root.title("Face Recognition Training System")

        # Title
        title_lbl = Label(self.root, text="TRAIN DATA SET", font=("times new roman", 35, "bold"), bg="white", fg="red")
        title_lbl.place(x=0, y=0, width=1530, height=45)

        # Top Image
        img_top = Image.open("collage_images/train.jpg")
        img_top = img_top.resize((1530, 325), Image.ANTIALIAS)
        self.photoimg_top = ImageTk.PhotoImage(img_top)

        f_lbl = Label(self.root, image=self.photoimg_top)
        f_lbl.place(x=0, y=55, width=1530, height=325)

        # Bottom Image
        img_bottom = Image.open("collage_images/train1.jpg")
        img_bottom = img_bottom.resize((1530, 325), Image.ANTIALIAS)
        self.photoimg_bottom = ImageTk.PhotoImage(img_bottom)

        f_lbl = Label(self.root, image=self.photoimg_bottom)
        f_lbl.place(x=0, y=440, width=1530, height=325)

        # Train Button
        b1_1 = Button(self.root, text="TRAIN DATA", command=self.train_classifier, cursor="hand2",
                      font=("times new roman", 30, "bold"), bg="red", fg="white")
        b1_1.place(x=0, y=380, width=1530, height=60)

    def train_classifier(self):
        try:
            data_dir = "data"  # Directory containing training images
            if not os.path.exists(data_dir):
                messagebox.showerror("Error", "Data directory not found!")
                return

            # Collect all image paths
            path = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith((".jpg", ".png"))]

            if not path:
                messagebox.showerror("Error", "No images found in the data directory!")
                return

            faces = []
            ids = []

            for image in path:
                try:
                    img = Image.open(image).convert('L')  # Convert to grayscale
                    imageNp = np.array(img, 'uint8')  # Convert image to NumPy array
                    id = int(os.path.split(image)[1].split('.')[1])  # Extract ID from filename

                    faces.append(imageNp)
                    ids.append(id)

                    # Display image during training
                    cv2.imshow("Training", imageNp)
                    cv2.waitKey(1)
                except Exception as e:
                    print(f"Error processing image {image}: {e}")

            ids = np.array(ids)

            # Train LBPH recognizer
            clf = cv2.face.LBPHFaceRecognizer_create()
            clf.train(faces, ids)
            clf.write("classifier.xml")  # Save trained model

            cv2.destroyAllWindows()
            messagebox.showinfo("Result", "Training dataset completed successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during training: {e}")


if __name__ == "__main__":
    root = Tk()
    obj = Train(root)
    root.mainloop()
