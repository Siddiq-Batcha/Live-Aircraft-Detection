import time
import cv2
import numpy as np
import math
import socket
import argparse
import os
import importlib.util
import customtkinter as ctk
from threading import Thread



# Define global flags

inner_flagA = False
mid_flagA = False

inner_flagB = False
mid_flagB = False

inner_flagC = False
mid_flagC = False



IP_Address_A = ""
IP_Address_B = ""
IP_Address_C = ""

##IP_A = 0 
##IP_B = 0
##IP_C = 0

Aflag = False
Bflag = False
Cflag = False

frameflagA = False
frameflagB = False
frameflagC = False

# Define video stream objects
videostream1 = None
videostream2 = None
videostream3 = None
flag = True


clcXA = 0
clcYA = 0


clcXB = 0
clcYB = 0


clcXC = 0
clcYC = 0

countL = 0
countA = 0
countB = 0
countC = 0


#ini cirlce radi







ymin = 0
xmin = 0
xmax = 0
ymax = 0


ymin2 = 0
xmin2 = 0
xmax2 = 0
ymax2 = 0


ymin3 = 0
xmin3 = 0
xmax3 = 0
ymax3 = 0

count =0

radius_intA = None
radiusA = None
Aera_RectangleA = None



radius_intB = None
radiusB = None
Aera_RectangleB = None



radius_intC = None
radiusC = None
Aera_RectangleC = None



outer_box_detected = False
inner_box_detected = False

oncounterA = 0
offcounterA = 0

oncounterB = 0
offcounterB = 0


oncounterC = 0
offcounterC = 0


countB = 0
countC = 0

num_dots = 24

# Calculate the angle increment between each dot
angle_increment = 360 / num_dots


imW= 0
imH = 0


center_xA = 0
center_yA = 0

center_xB = 0
center_yB = 0


center_xC = 0
center_yC = 0


host ="192.168.0.50"
port = 4444
UDP_SRC_PORT = 4445

HEADER = 0x11
LOCK_1 = 0x66


txbuff = bytearray(50)
lock = 1
lock_count = 0
error2deg = 0
error2deg_count = 0

Live_New_flag = False
Live_flag  = False
IFS_flag   = False
Night_flag = False

CircleA_flag = False
CircleB_flag = False
CircleC_flag = False

Live_Circle = False
IFS_Circle = False

Draw_flagA = False
Draw_flagB = False
Draw_flagC = False

Draw_IFSA = False
Draw_IFSB = False
Draw_IFSC = False


twodegA = False
twodegB = False
twodegC = False


detection_flag = 0
detection_flag2 = False

clickA_flag = False
clickB_flag = False
clickC_flag = False

clickA_IFS = False
clickB_IFS = False
clickC_IFS = False

Active_flagA = False
Active_flagB = False
Active_flagC = False
width = 0
length = 0

Black_Hot = False

def A_Key(key):
    global outer_circle_radius, inner_circle_radius, CircleA_flag,CircleB_flag,CircleC_flag, Live_Circle, IFS_Circle, Draw_flagA, Draw_flagB, Draw_flagC, Draw_IFSA, Draw_IFSB, Draw_IFSC, Live_Circle,IFS_Circle,sim_flag,Active_flagA,Active_flagB,Active_flagC
    if key == ord('a'):
        CircleA_flag = True
        CircleB_flag = False
        CircleC_flag = False
        sim_flag = True
        if Live_Circle == True:
            clickA_IFS = False
            Draw_flagA = True     
        if IFS_Circle == True:
            clickA_flag = False
            Draw_IFSA = True
         
        



def B_Key(key):
    global outer_circle_radius, inner_circle_radius, CircleA_flag,CircleB_flag,CircleC_flag, Live_Circle, IFS_Circle, Draw_flagA, Draw_flagB, Draw_flagC, Draw_IFSA, Draw_IFSB, Draw_IFSC
    if key == ord('b'):
        CircleA_flag = False
        CircleB_flag = True
        CircleC_flag = False
        if Live_Circle == True:
            clickB_IFS = False
##            Draw_flagA = False
            Draw_flagB = True
##            Draw_flagC = False
            Draw_IFSB = False
            
        if IFS_Circle == True:
            clickA_flag = False
            Draw_IFSB = True
            Draw_flagB = False



def C_Key(key):
    global outer_circle_radius, inner_circle_radius, CircleA_flag,CircleB_flag,CircleC_flag, Live_Circle, IFS_Circle, Draw_flagA, Draw_flagB, Draw_flagC, Draw_IFSA, Draw_IFSB, Draw_IFSC
    if key == ord('c'):
        CircleA_flag = False
        CircleB_flag = False
        CircleC_flag = True
        if Live_Circle == True:
            clickC_IFS = False
##            Draw_flagA = False
##            Draw_flagB = False
            Draw_flagC = True
            Draw_IFSC = False
            
        if IFS_Circle == True:
            clickA_flag = False
            Draw_IFSC = True
            Draw_flagC = False
        








def send_UDP(host, port,data_send):
    udp_socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    destination = (host,port)

    try:
        udp_socket.sendto(data_send.encode(), (host, port))
        print(f"data sent success to {host}:{port}")
    except Exception as e:
        print(f"Error sending data{e}")
    finally:
        udp_socket.close()


def print_debug_hex_string(header, data_buf, length):
    log_info = f"{header} : "
    for idx in range(length):
        log_info += f"{data_buf[idx]:02X},"
    print(log_info)

def send_udp_packet(data):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_socket:
        udp_socket.sendto(data, (host, port))




def Live_model_New():
    global outer_circle_radius, inner_circle_radius, Live_flag, IFS_flag, Night_flag, Live_Circle, IFS_Circle,Live_, Live_New_flag, width, length, Black_Hot
    Live_flag = True
    IFS_flag  = False
    Night_flag = False
    Live_Circle = True
    IFS_Circle = False
    Live_New_flag = True
    Black_Hot = False
    inner_circle_radius  = 240
    outer_circle_radius  = 310
    width = 1920
    length = 1080
    print("Live_New")
    






def Live_model():
    global outer_circle_radius, inner_circle_radius, Live_flag, IFS_flag, Night_flag, Live_Circle, IFS_Circle, Live_New_flag, width, length, Black_Hot
    Live_flag = True
    IFS_flag  = False
    Night_flag = False
    Live_Circle = True
    IFS_Circle = False
    Live_New_flag = False
    Black_Hot = False
    inner_circle_radius  = 240
    outer_circle_radius  = 310
    width = 1920
    length = 1080
    print("Live")

def IFS_model():
    global outer_circle_radius, inner_circle_radius, Live_flag, IFS_flag, Night_flag, Live_Circle, IFS_Circle, Live_New_flag, width, length, Black_Hot
    Live_flag = False
    IFS_flag  = True
    Night_flag = False
    Live_Circle = False
    IFS_Circle = True
    Live_New_flag = False
    Black_Hot = False
    inner_circle_radius  = 240
    outer_circle_radius  = 310
    width = 1920
    length = 1080
    print("IFS")

def Night_model():
    global outer_circle_radius, inner_circle_radius, Live_flag, IFS_flag, Night_flag, Live_Circle, IFS_Circle, Live_New_flag, width, length, Black_Hot
    Live_flag = False
    IFS_flag  = False
    Night_flag = True
    Live_Circle = False
    IFS_Circle = True
    Live_New_flag = False
    Black_Hot = False
    inner_circle_radius  = 50
    outer_circle_radius  = 100
    width = 640
    length = 480
    print("Night")



def Black_Model():
    global outer_circle_radius, inner_circle_radius, Live_flag, IFS_flag, Night_flag, Live_Circle, IFS_Circle, Live_New_flag, width, length, Black_Hot
    Live_flag = False
    IFS_flag  = False
    Night_flag = False
    Live_Circle = False
    IFS_Circle = True
    Live_New_flag = False
    inner_circle_radius  = 50
    outer_circle_radius  = 100
    width = 640
    length = 480
    Black_Hot = True
    print("Night")
    

def Start_Cam():
    global outer_circle_radius, inner_circle_radius, IP_Address_A, IP_Address_B, IP_Address_C, saved_textA, saved_textB, saved_textC, url,videostream1, Aflag, IP_A,IP_B,IP_C, Live_New_flag
    url = 0
    Aflag = True
    print(Aflag)
    



def while_loopA():
    global outer_circle_radius, inner_circle_radius,clcXA, clcYA, countA, outer1_circle_radius,outer_circle_radius,inner_circle_radius, Aflag, frameflagA,Aera_RectangleA,radiusA, radius_intA, flag, outer_box_detected,inner_box_detected,inner_radius,inner_flagA,mid_flagA, count, oncounterA,oncounterB,oncounterC,offcounterA,offcounterB,offcounterC,num_dots,angle_increment, Live_flag,IFS_flag,Night_flag,imW, imH, CircleA_flag, CircleB_flag, CircleC_flag, clcXB, clcYB, clcXC, clcYC, countB, countC, count, drag, center_yA,center_xA, center_yB,center_xB,center_yC,center_xC,xmin,ymin,xmax,ymax , Live_Circle, IFS_Circle, Draw_flagA, Draw_flagB, Draw_flagC, Draw_IFSA, Draw_IFSB, Draw_IFSC, twodegA,twodegB, twodegC,detection_flag,EnterKey, clickA_flag, clickB_flag, clickC_flag,clickA_IFS , clickB_IFS, clickC_IFS, Draw_IFSA,Draw_IFSB,Draw_IFSC, width, length, Black_Hot
    
    if Live_flag == True:
        print("Live Model True")
        parser = argparse.ArgumentParser()
        parser.add_argument('--modeldir', help='Folder the .tflite file is located in', default='C:\\POEMS_LIVE_IFS-0\\Live_Target')
        parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite', default='detect.tflite')
        parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt', default='labelmap.txt')
        parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects', default=0.5)
        parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.', default='1920x1080')
        parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection', action='store_true')

    elif IFS_flag == True:
        print("IFS Model True")
        parser = argparse.ArgumentParser()
        parser.add_argument('--modeldir', help='Folder the .tflite file is located in', default='C:\\POEMS_LIVE_IFS-0\\IFS_Target and Ball(New)')
        parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite', default='detect.tflite')
        parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt', default='labelmap.txt')
        parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects', default=0.5)
        parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.', default='1920x1080')
        parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection', action='store_true')

    elif Night_flag == True:
        print("Night Model True")
        parser = argparse.ArgumentParser()
        parser.add_argument('--modeldir', help='Folder the .tflite file is located in', default='C:\\POEMS_LIVE_IFS-0\\TI_LIVE_New')
        parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite', default='detect.tflite')
        parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt', default='labelmap.txt')
        parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects', default=0.4)
        parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.', default='640x480')
        parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection', action='store_true')

        
    elif Live_New_flag == True:
        print("Live New Model True")
        parser = argparse.ArgumentParser()
        parser.add_argument('--modeldir', help='Folder the .tflite file is located in', default='C:\\POEMS_LIVE_IFS-0\\Live_Target(New)')
        parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite', default='detect.tflite')
        parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt', default='labelmap.txt')
        parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects', default=0.4)
        parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.', default='1920x1080')
        parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection', action='store_true')


    elif Black_Hot == True:
        print("Night Black Model True")
        parser = argparse.ArgumentParser()
        parser.add_argument('--modeldir', help='Folder the .tflite file is located in', default='C:\\POEMS_LIVE_IFS-0\\Black_Hot')
        parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite', default='detect.tflite')
        parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt', default='labelmap.txt')
        parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects', default=0.4)
        parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.', default='640x480')
        parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection', action='store_true')

        

    args = parser.parse_args()

    MODEL_NAME = args.modeldir
    GRAPH_NAME = args.graph
    LABELMAP_NAME = args.labels
    min_conf_threshold = float(args.threshold)
    resW, resH = args.resolution.split('x')

    imW, imH = int(resW), int(resH)

    use_TPU = args.edgetpu


    pkg = importlib.util.find_spec('tflite_runtime')
    if pkg:
        from tflite_runtime.interpreter import Interpreter
    else:
        from tensorflow.lite.python.interpreter import Interpreter

    # If using Edge TPU, assign filename for Edge TPU model
    if use_TPU:
        if GRAPH_NAME == 'detect.tflite':
            GRAPH_NAME = 'edgetpu.tflite'

    # Get path to the current working directory
    CWD_PATH = os.getcwd()

    # Path to .tflite file and label map file
    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)
    PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

    # Load the label map
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Load the TensorFlow Lite model
    if use_TPU:
        interpreter = Interpreter(model_path=PATH_TO_CKPT, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    else:
        interpreter = Interpreter(model_path=PATH_TO_CKPT)

    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    floating_model = (input_details[0]['dtype'] == np.float32)
    input_mean = 127.5
    input_std = 127.5

    # Check output layer name to determine if this model was created with TF2 or TF1
    outname = output_details[0]['name']
    if 'StatefulPartitionedCall' in outname:
        boxes_idx, classes_idx, scores_idx = 1, 3, 0
    else:
        boxes_idx, classes_idx, scores_idx = 0, 1, 2
    # Initialize frame rate calculation
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    class VideoStream:
        def __init__(self, resolution=(width, length), framerate=30, source = None):
            if source is None:
                source = 0 
            self.stream = cv2.VideoCapture(source)
            ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            ret = self.stream.set(3, resolution[0])
            ret = self.stream.set(4, resolution[1])
            self.grabbed, self.frame = self.stream.read()
            self.stopped = False

        def start(self):
            Thread(target=self.update, args=()).start()
            return self

        def update(self):
            while True:
                if self.stopped:
                    self.stream.release()
                    return
                self.grabbed, self.frame = self.stream.read()

        def read(self):
            return self.frame

        def stop(self):
            self.stopped = True



   

    # Initialize video stream
    if Aflag == True:
        videostream1 = VideoStream(resolution=(imW, imH), framerate=30, source=url).start()
        frameflagA = True
##        print("framFlag is True")


    def click_eventA(event, xA, yA, flags, param):
        global clcXA, clcYA, countA, clcXB, clcYB, clcXC, clcYC, countB, countC, count, countL,drag, Draw_flagA, Draw_flagB, Draw_flagC, clickA_flag, clickB_flag, clickC_flag ,clickA_IFS , clickB_IFS, clickC_IFS,Draw_IFSA,Draw_IFSB,Draw_IFSC
        
            
      
        if event == cv2.EVENT_RBUTTONDOWN:
            if Draw_flagA == True:
                countA = countA+1
                clickA_flag = True
                clcXA, clcYA = xA, yA
                print("Clicked at (Ax={}, Ay={})".format(xA, yA))
                Draw_flagA = False
            if Draw_flagB == True:
                countB = countB + 1
            
                clickB_flag = True
                clcXB, clcYB = xA,yA
                Draw_flagB = False
            if Draw_flagC == True:
                countC = countC + 1
            
                clickC_flag = True
                clcXC, clcYC = xA,yA
                Draw_flagC = False


            if Draw_IFSA == True:
                countA = countA+1
                clickA_IFS = True
                clcXA, clcYA = xA, yA
                print("Clicked at (Ax={}, Ay={})".format(xA, yA))
                Draw_IFSA = False
            if Draw_IFSB == True:
                countB = countB + 1
            
                clickB_IFS = True
                clcXB, clcYB = xA,yA
                Draw_IFSB = False
            if Draw_IFSC == True:
                countC = countC + 1
            
                clickC_IFS = True
                clcXC, clcYC = xA,yA
                Draw_IFSC = False



    while Aflag:
        key = cv2.waitKey(1) & 0xFF
##        EnterKey = cv2.waitKey(1) & 0xFF
        if key == ord('s') and EnterKey == 13:
            print("Enter ")
##        print("in While A flag")
        if flag == True:
##            print("Flag True")
            HEADER = 0x11
            LOCK_1 = 0x66
            lock = 1
            error2deg = 1
        
            txbuff[0] = HEADER
            txbuff[1] = LOCK_1
            txbuff[2] = lock
            txbuff[3] = error2deg

        # Print debug information
            print_debug_hex_string("SENT", txbuff, 20)

        # Send the UDP packet
            send_udp_packet(bytes(txbuff))
        else:
            flag = True
            degflag = True
        
        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

        if Aflag == True:
##            print("Aflag is true")
        # Grab frame from video stream
            frame1 = videostream1.read()
            frame = frame1.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (width, height))
            input_data = np.expand_dims(frame_resized, axis=0)

            if floating_model:
                input_data = (np.float32(input_data) - input_mean) / input_std
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()

                # Retrieve detection results
                boxes_detected = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
                classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
                scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]
            else:
                print("No box ")

            if clickA_flag == True:
                cv2.circle(frame1,(clcXA, clcYA),inner_circle_radius,(0,255,255),2)
                cv2.circle(frame1,(clcXA, clcYA),outer_circle_radius,(255,0,0),2)
               
            if clickB_flag == True:
                cv2.circle(frame1,(clcXB, clcYB),inner_circle_radius,(0,165,255),2)
                cv2.circle(frame1,(clcXB, clcYB),outer_circle_radius,(0,165,255),2)
                
            if clickC_flag == True:
                cv2.circle(frame1,(clcXC, clcYC),inner_circle_radius,(0,255,255),2)
                cv2.circle(frame1,(clcXC, clcYC),outer_circle_radius,(0,255,255),2)
                
            
            if IFS_Circle == True:
                # Draw the dotted circle
                for i in range(num_dots):
                    angle = math.radians(i * angle_increment)

                    if clickA_IFS == True:
                        xA = int(clcXA + (inner_circle_radius  )* math.cos(angle))
                        yA = int(clcYA + (inner_circle_radius  ) * math.sin(angle))
                        xA1 = int(clcXA + (outer_circle_radius )* math.cos(angle))
                        yA1 = int(clcYA + (outer_circle_radius) * math.sin(angle))
                        cv2.circle(frame1, (xA1, yA1), 1, (255,0,0), 1)
                        cv2.circle(frame1, (xA, yA), 2, (255,0,0), 1)

                    if clickB_IFS == True:
                        xB = int(clcXB + (inner_circle_radius  )* math.cos(angle))
                        yB = int(clcYB + (inner_circle_radius  ) * math.sin(angle))
                        xB1 = int(clcXB + (outer_circle_radius )* math.cos(angle))
                        yB1 = int(clcYB + (outer_circle_radius) * math.sin(angle))
                        cv2.circle(frame1, (xB1, yB1), 1, (0,0,255), 1)
                        cv2.circle(frame1, (xB, yB), 2, (0,0,255), 1)


                    if clickC_IFS == True:
                        xC = int(clcXC + (inner_circle_radius  )* math.cos(angle))
                        yC = int(clcYC + (inner_circle_radius  ) * math.sin(angle))
                        xC1 = int(clcXC + (outer_circle_radius )* math.cos(angle))
                        yC1 = int(clcYC + (outer_circle_radius) * math.sin(angle))
                        cv2.circle(frame1, (xC1, yC1), 1, (0,255,255), 1)
                        cv2.circle(frame1, (xC, yC), 2, (0,255,255), 1)

            
            
            


            for i in range(len(scores)):
                if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                    detection_flag2 = True
##                    print(detection_flag2)
                    detection_flag = 0
                    ymin = int(max(1, (boxes_detected[i][0] * imH)))
                    xmin = int(max(1, (boxes_detected[i][1] * imW)))
                    ymax = int(min(imH, (boxes_detected[i][2] * imH)))
                    xmax = int(min(imW, (boxes_detected[i][3] * imW)))

                   
                    cv2.rectangle(frame1, (xmin, ymin), (xmax, ymax), (255, 0, 255), 2)

                    center_xA = (xmin + xmax) // 2
                    center_yA = (ymin + ymax) // 2
                    

                    center_xB = (xmin + xmax) // 2
                    center_yB = (ymin + ymax) // 2

                    center_xC = (xmin + xmax) // 2
                    center_yC = (ymin + ymax) // 2


                    object_widthA = xmax - xmin
                    object_heightA = ymax - ymin


                    
                
             

            

            
                            
        
            

            
                    if CircleA_flag == True:
                        cv2.putText(frame1,'A',(1700,900),cv2.FONT_HERSHEY_COMPLEX,1, (0,165,255),2)
   
                        if (((center_xA - clcXA)**2 + (center_yA - clcYA)**2)** 0.5) <= inner_circle_radius:
                            print("Lock on - A")
                        
                            inner_flagA = True
                            mid_flagA = True
                            twodegA  = True
                            oncounterA = oncounterA+1
                            if oncounterA >= 1:

                                inner_box_detected = True
                                oncounteA = 2
                                offcounterA=0
                        else:
                            inner_flagA = False
                            offcounterA = offcounterA+1
                            if offcounterA >= 10:
                                inner_box_detected = False
                                offcounterA = 5
                                oncounterA = 0
                            
                            
                        if twodegA == True:

                            if (xmin < clcXA + outer_circle_radius and 
                                xmax > clcXA - outer_circle_radius and 
                                ymin < clcYA + outer_circle_radius and 
                                ymax > clcYA - outer_circle_radius and 
                                not (xmin < clcXA + inner_circle_radius and 
                                xmax > clcXA - inner_circle_radius and 
                                ymin < clcYA + inner_circle_radius and 
                                ymax > clcYA - inner_circle_radius)):
                                outer_box_detected = True
                                print("2deg - A")
                                                
                                                
                                            
                            else:
                                mid_flagA = False
                                outer_box_detected = False
                    if CircleB_flag == True:
                            cv2.putText(frame1,'B',(1700,900),cv2.FONT_HERSHEY_COMPLEX,1, (0,165,255),2)
                            print("B circle")
                            if (center_xB - clcXB)**2 + (center_yB - clcYB)**2 <= inner_circle_radius**2:
                                print("Lock on - B")
                                inner_box_detected = True
                                twodegB  = True
                                oncounterB = oncounterA+1
                                if oncounterB >= 1:
                                    inner_box_detected = True
                                    oncounteB = 2
                                    offcounterB=0
                            else:
                                inner_flagB = False
                                
                                offcounterB = offcounterB+1
                                if offcounterB >= 10:
                                    inner_box_detected = False
                                    offcounterB = 5
                                    oncounterB = 0
                        
                        

                            if twodegB == True:
                                if (xmin < clcXB + outer_circle_radius and 
                                    xmax > clcXB - outer_circle_radius and 
                                    ymin < clcYB + outer_circle_radius and 
                                    ymax > clcYB - outer_circle_radius and 
                                    not (xmin < clcXB + inner_circle_radius and 
                                    xmax > clcXB - inner_circle_radius and 
                                    ymin < clcYB + inner_circle_radius and 
                                    ymax > clcYB - inner_circle_radius)):
                                    outer_box_detected = True
                                    print("2deg - B")
                                                    
                                                    
                                                
                                else:
                                    outer_box_detected = False


                    if CircleC_flag == True:
                        cv2.putText(frame1,'C',(1700,900),cv2.FONT_HERSHEY_COMPLEX,1, (0,255,255),2)
                        print("C circle")
                        if (center_xC - clcXC)**2 + (center_yC - clcYC)**2 <= inner_circle_radius**2:
                            print("Lock on - C")
                            inner_box_detected = True
                            twodegC  = True
                            oncounterC = oncounterC+1
                            if oncounterC >= 1:
                                inner_box_detected = True
                                oncounteC = 2
                                offcounterC=0
                        else:
                            
                            offcounterC = offcounterA+1
                            if offcounterC >= 10:
                                inner_box_detected = False
                                offcounterC = 5
                                oncounterC = 0
                            
                            

                        if twodegC == True:
                            if (xmin < clcXC + outer_circle_radius and 
                                xmax > clcXC - outer_circle_radius and 
                                ymin < clcYC + outer_circle_radius and 
                                ymax > clcYC - outer_circle_radius and 
                                not (xmin < clcXC + inner_circle_radius and 
                                xmax > clcXC - inner_circle_radius and 
                                ymin < clcYC + inner_circle_radius and 
                                ymax > clcYC - inner_circle_radius)):
                                
                                outer_box_detected = True
                                print("2deg - C")
                                                
                                                
                                            
                            else:
                                mid_flagA = False
                                outer_box_detected = False
                else:
                    detection_flag = detection_flag + 1
##                    print(detection_flag)
                    if detection_flag >= 30:
                        detection_flag2 = False
##                        print("No Lock")
                        inner_box_detected = False
            

                


            if outer_box_detected and not inner_box_detected:
                
                flag = False
                print("2deg")
                HEADER = 0x11
                LOCK_1 = 0x66
                lock = 1
                error2deg = 0
                txbuff[0] = HEADER
                txbuff[1] = LOCK_1
                txbuff[2] = lock
                txbuff[3] = error2deg

                # Print debug information
                print_debug_hex_string("SENT", txbuff, 20)

                # Send the UDP packet
                send_udp_packet(bytes(txbuff))

            elif inner_box_detected and outer_box_detected and detection_flag2:
                flag = False
                print("Flag False")
                print("Lock on in and out")
                HEADER = 0x11
                LOCK_1 = 0x66
                lock = 0
                error2deg = 1
                txbuff[0] = HEADER
                txbuff[1] = LOCK_1
                txbuff[2] = lock
                txbuff[3] = error2deg

                # Print debug information
                print_debug_hex_string("SENT", txbuff, 20)
            
                 # Send the UDP packet
                send_udp_packet(bytes(txbuff))
            elif inner_box_detected and detection_flag2:
                flag = False
                print("Lock on in")
                HEADER = 0x11
                LOCK_1 = 0x66
                lock = 0
                error2deg = 1
                txbuff[0] = HEADER
                txbuff[1] = LOCK_1
                txbuff[2] = lock
                txbuff[3] = error2deg

                # Print debug information
##                print_debug_hex_string("SENT", txbuff, 20)
            
                # Send the UDP packet
                send_udp_packet(bytes(txbuff))
  
            
            
            A_Key(key)
            B_Key(key)
            C_Key(key)
             

            cv2.putText(frame1, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.namedWindow('Object detector1')
            cv2.setMouseCallback('Object detector1', click_eventA)
            cv2.imshow('Object detector1', frame1)

            t2 = cv2.getTickCount()
            time1 = (t2 - t1) / freq
            frame_rate_calc = 1 / time1
        

            # Press 'q' to quit
            if key == ord('q') or key == ord('Q') :
                cv2.destroyWindow("Object detector1")
                videostream1.stop()
                break

   
def main():
    global IP_Address_A, IP_Address_B, IP_Address_C , IP_A, IP_B, IP_C, Port_A, Port_B, Port_C
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")
    main_frame = ctk.CTk()
    main_frame.geometry("500x400")

    def on_Start_button_click():
        Start_Cam()
        # Start the while loop in a separate thread
        Thread(target=while_loopA).start()


    Start = ctk.CTkButton(master = main_frame, text="Start", command=on_Start_button_click)
    Start.pack(pady=10,padx=10)
    Start.place_forget() 
    Start.place(x=175, y=100)

    Live_TF = ctk.CTkButton(master = main_frame, text="LIVE", command=Live_model)
    Live_TF.pack(pady=10,padx=10)
    Live_TF.place_forget() 
    Live_TF.place(x=20, y=30)

    IFS_TF = ctk.CTkButton(master = main_frame, text="A/C MODEL", command=IFS_model)
    IFS_TF.pack(pady=10,padx=10)
    IFS_TF.place_forget() 
    IFS_TF.place(x=170, y=30) 

    Night_TF = ctk.CTkButton(master = main_frame, text="TI MODE-White Hot", command=Night_model)
    Night_TF.pack(pady=10,padx=10)
    Night_TF.place_forget() 
    Night_TF.place(x=320, y=30)

    Live_New = ctk.CTkButton(master = main_frame, text="LIVE_New", command=Live_model_New)
    Live_New.pack(pady=10,padx=10)
    Live_New.place_forget() 
    Live_New.place(x=20, y=80)

    Night_Black = ctk.CTkButton(master = main_frame, text="TI MODE-Black Hot", command=Black_Model)
    Night_Black.pack(pady=10,padx=10)
    Night_Black.place_forget() 
    Night_Black.place(x=320, y=80)



    main_frame.mainloop()

if __name__ == "__main__":
    main()
    


 
