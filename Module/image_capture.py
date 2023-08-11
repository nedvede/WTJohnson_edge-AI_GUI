from pypylon import pylon
import numpy as np
import cv2


def img_capture(shutter_speed = 800, totoal_img = 100):

    imageWindow = pylon.PylonImageWindow()
    imageWindow.Create(1)

    # Conecting to the available camera
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    #camera.MaxNumBuffer = 15

    camera.Width = 818   #最大4093
    camera.Height = 100    ##设为500

    # Number of grabbing images
    countOfImagesToGrab = 5
    # Exit code of the sample application
    exitCode = 0

    #camera.PixelFormat.SetValue('Mono8')
    #camera.AcquisitionMode.SetValue('Continuous')  # set continuing acquisition
    #camera.GainAuto.SetValue("Continuous")         # set Autogain
    #camera.ExposureAuto.SetValue("Continuous")     # set exposure time

    camera.ExposureTimeAbs.SetValue(shutter_speed) #328

    # Grabing Continusely (video) with minimal delay 
    camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
    converter = pylon.ImageFormatConverter()

    #Create an image size: (1000,4096)
    ###imgfile = numpy.ones((1000,4096))

    numCount = 0
    print('start')

    while camera.IsGrabbing():
        # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        
        # If successful grabbing, then
        if grabResult.GrabSucceeded():
        
            # Convert line grabbing image
            img = grabResult.Array 
            # img = cv2.resize(img, (818, 100)) 
            #img = 255 - img

            #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #test1:
            #img_cll = Line_conv.hsl_average(img)
            #Line_conv.weft_bias(img_cll = img_cll, line_num_true = 5, ro_col = 0, numCount = numCount)
            
            # #test2：
            # alignment = Line_conv.misalignment_detector(img, 200)
            # if alignment:
            #     #控制机器停止######################################
            #     mouse.position = (719,49)
            #     mouse.press(Button.left)
            #     mouse.release(Button.left)

            #     #mouse.move(-450,664)
            #     mouse.position = (236,723)
            #     mouse.press(Button.left)
            #     mouse.release(Button.left)
            #     #################################################

            #     print('Misalignment detected')
            #     cv2.imwrite("saved_reject-" + str(numCount) +".png", img)
            #     break
            # else:
            #     print('Qualified')
            #     cv2.imwrite("saved_accept-" + str(numCount) +".png", img)

            #cv2.imshow('title', img)      
            # cv2.waitKey(0)
            #cv2.imwrite("saved_pypylon_Febric_F-" + str(numCount) + ".png", img)

            # numCount += 1

            if numCount == totoal_img: # 500 lines forming a frame 
                break
            
                #Inspect_image = Ins.InspectA(numpy.uint8(imgfile))
                
        ###      cv2.namedWindow('title', cv2.WINDOW_NORMAL)
        ###      cv2.imshow('title', (numpy.uint8(imgfile)))
                # Save image
                

        ###      numCount = 0
        ###      Count += 1
        ###      print("Count #:", Count)

            
        ###      k = cv2.waitKey(1)
        ###       if k == 100:
                #  break
        grabResult.Release()

    # Releasing the resource
    camera.StopGrabbing()
    # Destroying windows
    #cv2.destroyAllWindows()
    # Close camera
    camera.Close()
    
    return img

