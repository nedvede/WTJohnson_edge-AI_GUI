import os
import glob
import shutil
from datetime import datetime
import time
import tkinter as tk
from tkinter import ttk, filedialog
import tkinter.messagebox as messagebox
from PIL import Image, ImageTk
import numpy as np
#from pypylon import pylon
from Module import Line_conv
from Module import find_skeleton, python_control_pico
from model import model_applicarion

class App:
    def __init__(self, root):
        self.root = root
        self.isRunning = False
        self.counter = 0
        self.num_images = tk.IntVar()
        self.speed = tk.DoubleVar()
        self.save_path_dir = tk.StringVar()
        ####manual_inspector变量######################
        self.inspector_window = None
        self.image_label = None
        self.image_source_path = tk.StringVar()
        self.image_save_path = tk.StringVar()
        self.image = None
        #######auto_inspector变量##################
        self.process_option = tk.StringVar()
        self.process_option.set("Medium")  # default value
        self.manual_check = tk.IntVar()
        self.process_check = tk.IntVar()
        self.isRunning = False


        # Manual Inspector button
        btn_manual = tk.Button(root, text="Image capture", command=self.open_inspector_window)
        btn_manual.pack(pady=10) # padding in the y direction
        lbl_manual = tk.Label(root, text="This button is used for capture images.")
        lbl_manual.pack(pady=5)

        # manual Inspector button
        btn_auto = tk.Button(root, text="Manual Inspector", command=self.manual_inspector)
        btn_auto.pack(pady=10)
        lbl_auto = tk.Label(root, text="This button is used for auto inspection.")
        lbl_auto.pack(pady=5)

        # Auto Inspector button
        btn_auto = tk.Button(root, text="Auto Inspector", command=self.auto_inspector)
        btn_auto.pack(pady=10)
        lbl_auto = tk.Label(root, text="This button is used for auto inspection.")
        lbl_auto.pack(pady=5)

        # Exit button
        btn_auto = tk.Button(root, text="Exit", command=self.exit_app)
        btn_auto.pack(pady=10)
        lbl_auto = tk.Label(root, text="This button is used for close the software.")
        lbl_auto.pack(pady=5)
###########################################################################################
#############exit_app函数定义###############################################################
    def exit_app(self):
        self.root.destroy()

###########################################################################################
#############image_capture函数定义###############################################################

    def open_inspector_window(self):
        self.new_window = tk.Toplevel(self.root)
        self.new_window.geometry('1400x500') # Window size: 820x500 pixels
        
        # Configure the rows and columns
        for i in range(6):  
            self.new_window.grid_rowconfigure(i, weight=1)
        for i in range(3):
            self.new_window.grid_columnconfigure(i, weight=1)

        # Create a black image of size 100x818
        self.image_original = Image.fromarray(np.zeros((150, 1227), dtype=np.uint8))
        self.tk_image_original = ImageTk.PhotoImage(self.image_original)

        self.image_original_label = tk.Label(self.new_window, image=self.tk_image_original,  width=1227, height=150, pady=2)
        self.image_original_label.grid(row=0, column=0, columnspan=3)

        self.counter_label = tk.Label(self.new_window, text="Number of captured images: 0")
        self.counter_label.grid(row=1, column=0, columnspan=3)

        save_path = tk.Label(self.new_window, text="Save path setting:")
        save_path.grid(row=2, column=0)
        save_path_entry = tk.Entry(self.new_window, textvariable=self.save_path_dir)
        save_path_entry.grid(row=2, column=1)

        speed_label = tk.Label(self.new_window, text="Shutter Speed (ms):")
        speed_label.grid(row=3, column=0)
        speed_entry = tk.Entry(self.new_window, textvariable=self.speed)
        speed_entry.grid(row=3, column=1)

        num_images_label = tk.Label(self.new_window, text="Set number of images to capture:")
        num_images_label.grid(row=4, column=0)
        num_images_entry = tk.Entry(self.new_window, textvariable=self.num_images)
        num_images_entry.grid(row=4, column=1)

        self.progress = ttk.Progressbar(self.new_window, length=600, mode='determinate', maximum=100)
        self.progress.grid(row=5, column=0, columnspan=3)

        btn_start = tk.Button(self.new_window, text="Start", command=self.start_inspection, width=20, height=1)
        btn_start.grid(row=6, column=0)

        btn_stop = tk.Button(self.new_window, text="Stop", command=self.stop_inspection, width=20, height=1)
        btn_stop.grid(row=6, column=1)

        btn_exit = tk.Button(self.new_window, text="Exit", command=self.exit_inspection1, width=20, height=1)
        btn_exit.grid(row=6, column=2)


    def manual_inspect_pattern(self):
        if self.isRunning:
            imageWindow = pylon.PylonImageWindow()
            imageWindow.Create(1)

            # Conecting to the available camera
            self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            self.camera.Open()

            self.camera.Width = 4093 # 4092  
            self.camera.Height = 500 #500    

            self.camera.ExposureTimeAbs.SetValue(self.speed.get()) 

            # Grabing Continusely (video) with minimal delay 
            self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
            converter = pylon.ImageFormatConverter()

            print('start')
            if self.camera.IsGrabbing():

                # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
                grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

                # If successful grabbing, then
                if grabResult.GrabSucceeded() and self.isRunning:

                    # Convert line grabbing image
                    img = grabResult.Array 
                    img = np.flip(img, axis=0)
                    #img_reshape = np.resize(img, (150, 1227))

                    self.image_original = Image.fromarray(img).resize((150, 1227))

                    self.save_image(self.save_path_dir.get())

                    self.tk_image_original = ImageTk.PhotoImage(self.image_original)
                    self.image_original_label.config(image=self.tk_image_original)
                    self.counter += 1
                    print(self.counter)
                    self.counter_label.config(text=f"Number of captured images: {self.counter}")

                    self.progress['value'] = (self.counter / self.num_images.get()) * 100  # Update the progress bar

                    if self.counter == self.num_images.get(): 
                        stop_output = python_control_pico.execute_ampy_command("ampy --port COM3 run C:/Line_detection_leeds/pico/test.py")
                        print("end")
                        self.isRunning = False

                        # Show the message
                        messagebox.showinfo("Information", "Image capture finished")
                    else:
                        self.root.after(300, self.manual_inspect_pattern)  # Continue capturing images after a delay

                grabResult.Release()

        if not self.isRunning:
            # Releasing the resource
            self.camera.StopGrabbing()
            # Close camera
            self.camera.Close() 

    def save_image(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S") # current time in YYYYMMDD-HHMMSS format
        filename = f"image_{current_time}.jpg"  # create filename
        img_path = os.path.join(folder, filename)  # create path
        self.image_original.save(img_path)

    
    def start_inspection(self):
        self.isRunning = True
        if self.speed.get() < 1 or self.num_images.get()<1:
            messagebox.showinfo("Manual inspect", "Please input valid shutter speed or image number.")
        else:
            self.manual_inspect_pattern()

    def stop_inspection(self):
        stop_output = python_control_pico.execute_ampy_command("ampy --port COM3 run C:/Line_detection_leeds/pico/test.py")
        print("end")
        self.isRunning = False

    def exit_inspection1(self):
        self.new_window.destroy()
##########################################################################################
##################Manual inspect函数定义##################################################
    def manual_inspector(self):
        self.inspector_window = tk.Toplevel(self.root)
        self.inspector_window.geometry('1400x500')

        # Create a black image of size 150x1227
        self.image = Image.fromarray(np.zeros((150, 1227), dtype=np.uint8))
        self.tk_image = ImageTk.PhotoImage(self.image)

        self.image_label = tk.Label(self.inspector_window, image=self.tk_image)
        self.image_label.grid(row=0, column=0, columnspan=2, sticky="nsew")

        tk.Label(self.inspector_window, text="Image Source Path:").grid(row=1, column=0, sticky="w")
        self.browse_source_button = tk.Button(self.inspector_window, text="Browse", command=self.browse_source)
        self.browse_source_button.grid(row=1, column=1, sticky="w")
        
        tk.Label(self.inspector_window, text="Image Save Path:").grid(row=3, column=0, sticky="w")
        self.browse_save_button = tk.Button(self.inspector_window, text="Browse", command=self.browse_save)
        self.browse_save_button.grid(row=3, column=1, sticky="w")

        self.accept_button = tk.Button(self.inspector_window, text="Accept", command=self.accept_image)
        self.accept_button.grid(row=5, column=0, sticky="nsew")
        self.accept_button.config(state='disabled')

        self.reject_button = tk.Button(self.inspector_window, text="Reject", command=self.reject_image)
        self.reject_button.grid(row=5, column=1, sticky="nsew")
        self.reject_button.config(state='disabled')

        self.start_button = tk.Button(self.inspector_window, text="Start", command=self.start_loading)
        self.start_button.grid(row=6, column=0, sticky="nsew")

        self.exit_button = tk.Button(self.inspector_window, text="Exit", command=self.inspector_window.destroy)
        self.exit_button.grid(row=6, column=1, sticky="nsew")

        # Configure the rows and columns so that they adjust as the window size changes.
        for i in range(6):
            self.inspector_window.grid_rowconfigure(i, weight=1)
            for j in range(2):
                self.inspector_window.grid_columnconfigure(i, weight=1)

    def browse_source(self):
        source_path = filedialog.askdirectory()
        self.image_source_path.set(source_path)
        tk.Label(self.inspector_window, text=self.image_source_path.get()).grid(row=2, column=0, columnspan=2, sticky="w")


    def browse_save(self):
        save_path = filedialog.askdirectory()
        self.image_save_path.set(save_path)
        tk.Label(self.inspector_window, text=self.image_save_path.get()).grid(row=4, column=0, columnspan=2, sticky="w")

    def accept_image(self):
        self.move_image('accept')
        self.display_next_image()

    def reject_image(self):
        self.move_image('reject')
        self.display_next_image()

    def start_loading(self):
        if (self.image_save_path.get() == "") or (self.image_source_path.get() == ""):
            messagebox.showinfo("Manual Inspector", "Please set the path.")
        else:
            self.accept_button.config(state='normal')
            self.reject_button.config(state='normal')

            # Load list of image paths
            self.images_list = glob.glob(os.path.join(self.image_source_path.get(), '*.png'))

            # Reset the image index
            self.image_index = 0

            # Display the first image
            self.display_next_image()

    def display_next_image(self):
        if self.image_index < len(self.images_list):
            # Load image
            self.image = Image.open(self.images_list[self.image_index])
            self.image_original = self.image.resize((1227, 150))

            # Update the image in the label
            self.tk_image = ImageTk.PhotoImage(self.image_original)
            self.image_label.config(image=self.tk_image)

            # Increment the index
            # self.image_index += 1
            print(self.image_index)
        else:
            messagebox.showinfo("manual Inspector", "All of the image has been anotated.")

    def move_image(self, directory):
        move_path = os.path.join(self.image_save_path.get(), directory)
        os.makedirs(move_path, exist_ok=True)

        source_path = os.path.join(self.image_source_path.get(), self.images_list[self.image_index])
        # Move the file
        shutil.move(source_path, move_path)
        self.image_index += 1
##############################################################################################
###################Auto_inspector###############################################
    def auto_inspector(self):
        self.auto_window = tk.Toplevel(self.root)
        self.auto_window.geometry('1300x500')

        # Status Indicator
        self.status_indicator1 = tk.Label(self.auto_window, text="STOP", bg="gray", width=10, height=1)
        self.status_indicator1.grid(row=0, column=0)
       

        self.status_indicator2 = tk.Label(self.auto_window, text="normal", bg="green", width=10, height=1)
        self.status_indicator2.grid(row=0, column=2)

        # Create a black image of size 150x1227
        self.image = Image.fromarray(np.zeros((150, 1227), dtype=np.uint8))
        self.tk_image = ImageTk.PhotoImage(self.image)

        self.image_label = tk.Label(self.auto_window, image=self.tk_image)
        self.image_label.grid(row=1, column=0, columnspan=3, sticky="nsew")

        self.counter_label = tk.Label(self.auto_window, text="Number of captured images: 0")
        self.counter_label.grid(row=2, column=0, columnspan=3)

        self.sensi = tk.Label(self.auto_window, text="Sensitivity:")
        self.sensi.grid(row=3, column=0)
        self.option_menu = tk.OptionMenu(self.auto_window, self.process_option, "Low", "Medium", "High")
        self.option_menu.grid(row=3, column=1)

        save_path = tk.Label(self.auto_window, text="Save path setting:")
        save_path.grid(row=4, column=0)
        save_path_entry = tk.Entry(self.auto_window, textvariable=self.save_path_dir)
        save_path_entry.grid(row=4, column=1)

        speed_label = tk.Label(self.auto_window, text="Shutter Speed (ms):")
        speed_label.grid(row=5, column=0)
        speed_entry = tk.Entry(self.auto_window, textvariable=self.speed)
        speed_entry.grid(row=5, column=1)

        num_images_label = tk.Label(self.auto_window, text="Set maximum of images to capture:")
        num_images_label.grid(row=6, column=0)
        num_images_entry = tk.Entry(self.auto_window, textvariable=self.num_images)
        num_images_entry.grid(row=6, column=1)

        self.accept_button = tk.Button(self.auto_window, text="Accept", command=self.accept_result, width=30, height=1)
        self.accept_button.grid(row=7, column=0)
        self.accept_button.config(state='disabled')

        self.reject_button = tk.Button(self.auto_window, text="Reject", command=self.reject_result, width=30, height=1)
        self.reject_button.grid(row=7, column=2)
        self.reject_button.config(state='disabled')

        self.start_button = tk.Button(self.auto_window, text="Start", command=self.start_detection, width=20, height=1)
        self.start_button.grid(row=8, column=0)

        self.stop_button = tk.Button(self.auto_window, text="Stop", command=self.stop_detection, width=20, height=1)
        self.stop_button.grid(row=8, column=1)

        self.exit_button = tk.Button(self.auto_window, text="Exit", command=self.auto_window.destroy, width=20, height=1)
        self.exit_button.grid(row=8, column=2)

        # Configure the rows and columns so that they adjust as the window size changes.
        for i in range(8):
            self.auto_window.grid_rowconfigure(i, weight=1)
            for j in range(3):
                self.auto_window.grid_columnconfigure(i, weight=1)

    def auto_inspect_pattern(self):
        if self.isRunning:
            imageWindow = pylon.PylonImageWindow()
            imageWindow.Create(1)

            # Conecting to the available camera
            self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            self.camera.Open()

            self.camera.Width = 1227
            self.camera.Height = 150    

            self.camera.ExposureTimeAbs.SetValue(self.speed.get()) 

            # Grabing Continusely (video) with minimal delay 
            self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
            converter = pylon.ImageFormatConverter()

            print('start')
            if self.camera.IsGrabbing():

                # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
                grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

                # If successful grabbing, then
                if grabResult.GrabSucceeded() and self.isRunning:

                    # Convert line grabbing image
                    img = grabResult.Array 
                    # img = np.flip(img, axis=0)
                    # img_reshape = np.resize(img, (150,1227))

                    self.image_original = Image.fromarray(img).resize((150, 1227))
                    

                    # self.save_image(self.save_path_dir.get())

                    process_algorithm = self.process_option.get()

                    # Replace this with the actual resizing algorithm
                    if process_algorithm == 'Low':
                        misalingnment = 0
                    elif process_algorithm == 'Medium':
                        misalingnment = model_applicarion.model_application(self.image_original, model_name = "alexnet_grayscale")
                        # print(misalingnment)
                    elif process_algorithm == 'High':
                        misalingnment = 1
                    else:
                        misalingnment = 1

                    if misalingnment:
                        self.stop_algorithm()
                        self.status_indicator2.config(text="Misalignment", bg="red")
                        stop_output = python_control_pico.execute_ampy_command("ampy --port COM3 run C:/Line_detection_leeds/pico/test.py")

                        self.tk_image_original = ImageTk.PhotoImage(self.image_original)
                        self.image_label.config(image=self.tk_image_original)

                        self.accept_button.config(state='normal')
                        self.reject_button.config(state='normal')

                        # Show the message
                        messagebox.showinfo("Information", "Misalignment detect!")
                    
                    self.counter += 1
                    print(self.counter)
                    self.counter_label.config(text=f"Number of captured images: {self.counter}")

                    if self.counter == self.num_images.get(): 
                        stop_output = python_control_pico.execute_ampy_command("ampy --port COM3 run C:/Line_detection_leeds/pico/test.py")
                        print("end")
                        self.isRunning = False

                        # Show the message
                        messagebox.showinfo("Information", "Scan reachs maximum!")
                    else:
                        self.root.after(300, self.auto_inspect_pattern)  # Continue capturing images after a delay

                grabResult.Release()

        if not self.isRunning:
            # Releasing the resource
            self.camera.StopGrabbing()
            # Close camera
            self.camera.Close() 

    def stop_algorithm(self):
        self.status_indicator1.config(text="STOP", bg="gray")
        self.isRunning = False

    def accept_result(self):
        savedd_path = os.path.join(self.save_path_dir.get(), 'accept')
        self.save_result(self.image_original, savedd_path)
        print("save successfully")


    def reject_result(self):
        rejj_path = os.path.join(self.save_path_dir.get(), 'reject')
        self.save_result(self.image_original, rejj_path)
        print("save successfully")

    def save_result(self, img, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S") # current time in YYYYMMDD-HHMMSS format
        filename = f"image_{current_time}.jpg"  # create filename
        img_path = os.path.join(folder, filename)  # create path
        img.save(img_path)

    
    def start_detection(self):
        self.isRunning = True
        if self.speed.get() < 1 or self.num_images.get()<1:
            messagebox.showinfo("Manual inspect", "Please input valid shutter speed or image number.")
        else:
            self.auto_inspect_pattern()

    def stop_detection(self):
        stop_output = python_control_pico.execute_ampy_command("ampy --port COM3 run C:/Line_detection_leeds/pico/test.py")
        print("end")
        self.isRunning = False

    def exit_inspection(self):
        self.auto_window.destroy()


# Create main window
root = tk.Tk()
root.title('Home Menu')

app = App(root)

# Run the mainloop
root.mainloop()
