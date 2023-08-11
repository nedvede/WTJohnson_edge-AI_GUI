import cv2
from tkinter import Tk, Button, Label, OptionMenu, StringVar, Entry, Frame
from PIL import ImageTk, Image
import numpy as np
import os
from Module import image_capture

class ImageCaptureGUI:
    def __init__(self, master):
        self.master = master
        master.title("Image Capture GUI")

        self.detection_option = StringVar(master)
        self.detection_option.set("No")  # default value

        self.width = StringVar(master)
        self.height = StringVar(master)

        self.option_frame = Frame(master)
        self.option_frame.grid(row=1, column=0)

        self.resize_label = Label(self.option_frame, text="Resize ")
        self.resize_label.pack(side="left")
        self.option_menu = OptionMenu(self.option_frame, self.resize_option, "Yes", "No")
        self.option_menu.pack(side="left")

        self.width_label = Label(self.option_frame, text="Width: ")
        self.width_label.pack()
        self.width_entry = Entry(self.option_frame, textvariable=self.width)
        self.width_entry.pack()

        self.height_label = Label(self.option_frame, text="Height: ")
        self.height_label.pack()
        self.height_entry = Entry(self.option_frame, textvariable=self.height)
        self.height_entry.pack()

        self.misalign_button = Button(self.option_frame, text="Misalignment", command=self.save_misalignment)
        self.misalign_button.pack()

        self.qualify_button = Button(self.option_frame, text="Qualify", command=self.save_qualify)
        self.qualify_button.pack()

        self.image_frame = Frame(master)
        self.image_frame.grid(row=0, column=0)

        self.image_label = Label(self.image_frame)
        self.image_label.pack()

        self.capture_image()

    def capture_image(self):

        frame = image_capture.img_capture()
        ret = 1

        if not ret:
            print("Failed to capture image")
            frame = np.zeros((440, 1440), dtype=np.uint8)
        
        self.image = cv2.resize(frame, (800, 140))
        self.image = Image.fromarray(self.image)

        if self.resize_option.get() == "Yes":
            width = int(self.width.get())
            height = int(self.height.get())
            self.image = self.image.resize((width, height), Image.ANTIALIAS)

        self.tk_image = ImageTk.PhotoImage(self.image)
        self.image_label.config(image=self.tk_image)

    def save_image(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        img_path = os.path.join(folder, "captured_image.jpg")
        self.image.save(img_path)
        self.capture_image()

    def save_misalignment(self):
        self.save_image("misalignment")

    def save_qualify(self):
        self.save_image("qualify")


root = Tk()
my_gui = ImageCaptureGUI(root)
root.mainloop()
