import os
import glob
from tkinter import Tk, Button, Label
from PIL import ImageTk, Image

class ImageBrowser:
    def __init__(self, master, folder, ext="*.png"):
        self.master = master
        self.folder = folder
        self.ext = ext
        self.img_files = glob.glob(os.path.join(folder, ext))
        self.index = 0

        self.img_label = Label(master)
        self.img_label.pack()

        self.next_button = Button(master, text="Next", command=self.show_next_img)
        self.next_button.pack()

        self.show_img()

    def show_img(self):
        print('1', self.img_files)
        if self.index >= len(self.img_files):  # no more images
            self.img_label.config(image=None)
            self.img_label.image = None
            print("No more images.")
        else:
            img_path = self.img_files[self.index]
            img = Image.open(img_path)
            img = img.resize((250, 250), Image.ANTIALIAS)  # resizing image
            img = ImageTk.PhotoImage(img)
            self.img_label.config(image=img)
            self.img_label.image = img  # keep a reference!

    def show_next_img(self):
        if self.index < len(self.img_files) - 1:  # if there are more images
            self.index += 1
        self.show_img()

root = Tk()
img_browser = ImageBrowser(root, "/Users/yueshi/Library/CloudStorage/OneDrive-UniversityofLeeds/deeplearn/GUI1/images_folder")  # replace with your actual folder name
root.mainloop()
