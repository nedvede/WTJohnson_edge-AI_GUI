import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
import cv2


root = tk.Tk()
root.title("Armstrong Autonomous Moon Landing System")
root.geometry("1100x600")
root.iconbitmap('C:/Users/brett/')


##########  ALL OTHER CODE NEEDS TO GO HERE


def open():
    global my_image
    filename = filedialog.askopenfilename(initialdir="images", title="Select A File", filetypes=(("png files", "*.png"),("all files", "*.*")))
    my_label.config(text=filename)
    my_image = Image.open(filename)
    tkimg = ImageTk.PhotoImage(my_image)
    my_image_label.config(image=tkimg)
    my_image_label.image = tkimg  # save a reference of the image


def find_craters():
    # convert PIL image to OpenCV image
    circles_image = np.array(my_image.convert('RGB'))
    gray_img = cv2.cvtColor(circles_image, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(gray_img, 5)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=0, maxRadius=0)
    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0]:
            # draw the outer circle
            cv2.circle(circles_image, (i[0],i[1]), i[2], (0,255,0), 2)
            # draw the center of the circle
            cv2.circle(circles_image, (i[0],i[1]), 2, (0,0,255), 3)

        # convert OpenCV image back to PIL image
        image = Image.fromarray(circles_image)
        # update shown image
        my_image_label.image.paste(image)


tk.Button(root, text="Load Terrain", command=open).pack()
tk.Button(root, text="Find Craters", command=find_craters).pack()

# for the filename of selected image
my_label = tk.Label(root)
my_label.pack()

# for showing the selected image
my_image_label = tk.Label(root)
my_image_label.pack()

root.mainloop()