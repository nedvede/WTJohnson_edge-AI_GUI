import cv2
from tkinter import Tk, Button as tkButton, Label, OptionMenu, StringVar, Entry, Frame, Checkbutton, IntVar
from PIL import ImageTk, Image
import numpy as np
import os
from datetime import datetime
from pynput.mouse import Button, Controller
from Module import image_capture
from Module import Line_conv
from Module import find_skeleton, python_control_pico
from model import model_applicarion

class ImageCaptureGUI:
    def __init__(self, master):
        self.master = master
        master.title("Image Capture GUI")
        master.geometry("1200x300")

        # Configure the rows and columns
        for i in range(7):  
            master.grid_rowconfigure(i, weight=1)
        for i in range(2):
            master.grid_columnconfigure(i, weight=1)

        self.process_option = StringVar(master)
        self.process_option.set("Sensitivity")  # default value

        self.manual_check = IntVar()
        self.process_check = IntVar()
        
        self.isRunning = False

        # Status Indicator
        self.status_indicator1 = Label(master, text="STOP", bg="gray", width=10, height=1)
        self.status_indicator1.grid(row=0, column=0)
       

        self.status_indicator2 = Label(master, text="normal", bg="green", width=10, height=1)
        self.status_indicator2.grid(row=0, column=1)

        self.image_original_frame = Frame(master, width=40, height=5, pady=2)
        self.image_original_frame.grid(row=2, column=0, columnspan=2)

        self.image_original_label = Label(self.image_original_frame)
        self.image_original_label.pack()

        self.image_processed_frame = Frame(master, width=40, height=5, pady=2)
        self.image_processed_frame.grid(row=3, column=0, columnspan=2)

        self.image_processed_label = Label(self.image_processed_frame)
        self.image_processed_label.pack()

        self.manual_check_button = Checkbutton(master, text="Manual Inspect", variable=self.manual_check, width=20, height=1)
        self.manual_check_button.grid(row=4, column=0)

        self.check_button = Checkbutton(master, text="Misalignment Detection", variable=self.process_check, command=self.toggle_dropdown, width=20, height=1)
        self.check_button.grid(row=5, column=0)

        self.option_menu = OptionMenu(master, self.process_option, "Low", "Medium", "High")
        self.option_menu.grid(row=5, column=1)

        self.accept_button = tkButton(master, text="Accept", command=self.save_accept, width=20, height=1)
        self.accept_button.grid(row=6, column=0)

        self.reject_button = tkButton(master, text="Reject", command=self.save_reject, width=20, height=1)
        self.reject_button.grid(row=6, column=1)

        self.start_button = tkButton(master, text="Start", command=self.start_system, width=20, height=1)
        self.start_button.grid(row=7, column=0)

        self.stop_button = tkButton(master, text="Stop", command=self.stop_system, width=20, height=1)
        self.stop_button.grid(row=7, column=1)

        self.update_image()

    def update_image(self):
         if self.manual_check.get() == 0:
            self.close_orginal_open_processing()
         else:
            self.manual_inspect_pattern()

        # if self.isRunning:
        #     # capture image
        #     frame = image_capture.img_capture()
        #     ret = 1
        #     misalingnment = 0

        #     if not ret:
        #         print("Failed to capture image")
        #         frame = np.zeros((818, 100), dtype=np.uint8)
            
        #     self.image_original = frame
        #     self.image_original = Image.fromarray(self.image_original)
        #     self.image_processed = self.image_original # initialize with the original image

        #     if self.process_check.get() == 1:
        #         process_algorithm = self.process_option.get()

        #         # Replace this with the actual resizing algorithm
        #         if process_algorithm == 'Low':
        #             misalingnment = Line_conv.misalignment_detector(self.image_original, 40)
        #         elif process_algorithm == 'Medium':
        #             misalingnment = model_applicarion.model_application(self.image_original, model_name = "alexnet_grayscale")
        #             print(misalingnment)
        #         elif process_algorithm == 'High':
        #             misalingnment = Line_conv.misalignment_detector(self.image_original, 40)

        #     else:
        #         misalingnment = 0

        #     if misalingnment:
        #         self.stop_system()
        #         mouse = Controller()
        #         #控制机器停止######################################
        #         mouse.position = (1000,501)
        #         mouse.press(Button.left)
        #         mouse.release(Button.left)

        #         #mouse.move(-450,664)
        #         mouse.position = (255,718)
        #         mouse.press(Button.left)
        #         mouse.release(Button.left)
 
        #         skeleton = find_skeleton.image_line_detect(self.image_original)
        #         self.image_processed =  cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2RGB)
        #         print(np.shape(self.image_processed))
        #         self.image_processed[np.where(skeleton==255)] = [0,0,255]
        #         print(np.shape(self.image_processed))
        #         self.image_processed = Image.fromarray(np.uint8(self.image_processed))

        #         self.tk_image_processed = ImageTk.PhotoImage(self.image_processed)
        #         self.image_processed_label.config(image=self.tk_image_processed)

        #     self.tk_image_original = ImageTk.PhotoImage(self.image_original)
        #     self.image_original_label.config(image=self.tk_image_original)

            

        #     self.master.after(1000, self.update_image)  # update every second
    
    
    # def update_image(self):
    #     frame = image_capture.img_capture()
    #     ret = 1
    #     misalingnment = 0

    #     if not ret:
    #         print("Failed to capture image")
    #         frame = np.zeros((800, 140), dtype=np.uint8)
        
    #     self.image_original = cv2.process(frame, (800, 140))
    #     self.image_original = Image.fromarray(self.image_original)
    #     self.image_processed = self.image_original # initialize with the original image

    #     if self.process_check.get() == 1:
    #         process_algorithm = self.process_option.get()

    #         # For demonstration purposes only
    #         # Replace this with the actual resizing algorithm
    #         if process_algorithm == 'Low':
    #             misalingnment = Line_conv.misalignment_detector(self.image_original, 200)
    #         elif process_algorithm == 'Medium':
    #             # Apply resizing algorithm a2
    #             misalingnment = Line_conv.misalignment_detector(self.image_original, 200)
    #         elif process_algorithm == 'High':
    #             # Apply resizing algorithm a3
    #             misalingnment = Line_conv.misalignment_detector(self.image_original, 200)

    #     else:
    #         misalingnment = 0

    #     if misalingnment:
    #         mouse = Controller()
    #         #控制机器停止######################################
    #         mouse.position = (719,49)
    #         mouse.press(Button.left)
    #         mouse.release(Button.left)

    #         #mouse.move(-450,664)
    #         mouse.position = (236,723)
    #         mouse.press(Button.left)
    #         mouse.release(Button.left)

    #         skeleton = find_skeleton.image_line_detect(self.image_original)
    #         self.image_processed =  cv2.cvtColor(cv2.process(frame, (800, 140)), cv2.COLOR_BGR2RGB)
    #         self.image_processed[np.where(skeleton==255)] = [0,0,255]


    #     self.tk_image_original = ImageTk.PhotoImage(self.image_original)
    #     self.image_original_label.config(image=self.tk_image_original)

    #     self.tk_image_processed = ImageTk.PhotoImage(self.image_processed)
    #     self.image_processed_label.config(image=self.tk_image_processed)

    def manual_inspect_pattern(self):
        if self.isRunning:
            frame = image_capture.img_capture()
            ret = 1

            if not ret:
                print("Failed to capture image")
                frame = np.zeros((100, 818), dtype=np.uint8)

            self.image_original = frame
            self.image_original = Image.fromarray(self.image_original)
            self.image_processed = Image.fromarray(np.zeros((100, 818), dtype=np.uint8)) # initialize with the original image
    
            self.tk_image_original = ImageTk.PhotoImage(self.image_original)
            self.image_original_label.config(image=self.tk_image_original)
            self.tk_image_processed = ImageTk.PhotoImage(self.image_processed)
            self.image_processed_label.config(image=self.tk_image_processed)
    
    def close_orginal_open_processing(self):
        if self.isRunning:
            # capture image
            frame = image_capture.img_capture()
            ret = 1
            misalingnment = 0

            if not ret:
                print("Failed to capture image")
                frame = np.zeros((100, 818), dtype=np.uint8)
            
            self.image_original = frame
            self.image_original = Image.fromarray(self.image_original)
            self.image_processed = self.image_original # initialize with the original image

            if self.process_check.get() == 1:
                process_algorithm = self.process_option.get()

                # Replace this with the actual resizing algorithm
                if process_algorithm == 'Low':
                    misalingnment = Line_conv.misalignment_detector(self.image_original, 40)
                elif process_algorithm == 'Medium':
                    misalingnment = model_applicarion.model_application(self.image_original, model_name = "alexnet_grayscale")
                    print(misalingnment)
                elif process_algorithm == 'High':
                    misalingnment = Line_conv.misalignment_detector(self.image_original, 40)

            else:
                misalingnment = 0

            if misalingnment:
                self.stop_system()
                stop_output = python_control_pico.execute_ampy_command("ampy --port COM3 run C:/Line_detection_leeds/pico/test.py")
                print(stop_output)
                # mouse = Controller()
                # #控制机器停止######################################
                # mouse.position = (1000,501)
                # mouse.press(Button.left)
                # mouse.release(Button.left)

                # #mouse.move(-450,664)
                # mouse.position = (255,718)
                # mouse.press(Button.left)
                # mouse.release(Button.left)
 
                skeleton = find_skeleton.image_line_detect(self.image_original)
                self.image_processed =  cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2RGB)
                print(np.shape(self.image_processed))
                self.image_processed[np.where(skeleton==255)] = [0,0,255]
                print(np.shape(self.image_processed))
                self.image_processed = Image.fromarray(np.uint8(self.image_processed))

                self.tk_image_original = ImageTk.PhotoImage(self.image_original)
                self.image_original_label.config(image=self.tk_image_original)
                self.tk_image_processed = ImageTk.PhotoImage(self.image_processed)
                self.image_processed_label.config(image=self.tk_image_processed)
            
            else:
                self.blank = Image.fromarray(np.zeros((100, 818), dtype=np.uint8))

                self.tk_image_original = ImageTk.PhotoImage(self.image_original)
                self.image_original_label.config(image=self.tk_image_original)
                self.tk_image_processed = ImageTk.PhotoImage(self.blank)
                self.image_processed_label.config(image=self.tk_image_processed)

            self.master.after(1000, self.update_image)  # update every second

    def save_image(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S") # current time in YYYYMMDD-HHMMSS format
        filename = f"image_{current_time}.jpg"  # create filename
        img_path = os.path.join(folder, filename)  # create path
        self.image_original.save(img_path)

    def save_accept(self):
        self.save_image("accept")
        self.update_image()

    def save_reject(self):
        self.save_image("reject")
        self.update_image()

    def start_system(self):
        self.status_indicator1.config(text="RUNNING", bg="green")
        self.isRunning = True
        self.update_image()

    def stop_system(self):
        self.status_indicator1.config(text="STOP", bg="gray")
        self.status_indicator2.config(text="Misalignment", bg="red")
        self.isRunning = False

    def toggle_dropdown(self):
        if self.process_check.get() == 0:
            self.option_menu.config(state="disabled")
        else:
            self.option_menu.config(state="normal")


root = Tk()
my_gui = ImageCaptureGUI(root)
root.mainloop()
