from tkinter import *
import tkinter as tk
import os
import time
from tkinter import messagebox
from tkinter import filedialog
import PIL.Image, PIL.ImageTk
import cv2
import speed3


def on_close():
    close = messagebox.askokcancel("Close","Would you like to close the program?")
    if close:
        quit()
        root.destroy()
        sys.exit

class videoGUI:

    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        top_frame = Frame(self.window)
        top_frame.pack(side=TOP, pady=30, padx=100)
        

        bottom_frame = Frame(self.window)
        bottom_frame.pack(side=BOTTOM, pady=30,padx=100)

        self.window.width=1000
        self.window.height=1000
        self.window.resizable(width=False,height=False)

        # self.canvas = Canvas(top_frame)
        # self.canvas.pack()

        self.input_name = "init"
        self.output_name = "wow_final_output.avi"

        # Select Button
        self.btn_select=Button(bottom_frame, text="Select video file", width=15, command=self.open_file)
        self.btn_select.grid(row=0, column=0)

        # Record Button
        self.btn_rec=Button(bottom_frame, text="Record Video", width=15, command=self.startRec)
        self.btn_rec.grid(row=0, column=1)

        #start processing button
        self.btn_spro=Button(bottom_frame,text="Start Processing",width=15,command=self.startpro)
        self.btn_spro.grid(row=1)

        # self.btn_exit=B

        self.delay = 15   # ms
        self.window.iconbitmap('ouri.ico')
        self.window.mainloop()

    def startpro(self):
        y1 = 'y'
        y2 = 'y'
        if self.input_name!='init':
            Msgbox1 = tk.messagebox.askquestion('Show Processing Video','Do you want to see the video being processed?')
            if Msgbox1 == 'yes':
                y1 = 'n'
            Msgbox2 = tk.messagebox.askquestion('Save Graph','Do you want to save graph plotting speed vs time ?')
            if Msgbox2 == 'no':
                y2 = 'n'
            print(self.input_name)
            print(self.output_name)
            self.window.destroy()
            if speed3.functionCall(self.input_name,self.output_name,y1,y2) == 1:
                print('Processing Finished..')
                file_name = os.path.basename(self.output_name)
                index_of_dot = file_name.index('.') 
                file_name_without_extension = file_name[:index_of_dot]
                nmm = file_name_without_extension+'.avi'
                pathname = os.path.split(self.output_name)[0]
                pathname = pathname+'\\'+file_name_without_extension
                root1 = tk.Tk()
                mess = 'For Output Video and Graphs, please go to current directory of the script. Output file name will be '+nmm+' and graphs will be availabe in folder '+pathname
                tk.messagebox.showwarning('Processing Over',mess)
                root1.destroy()
            else:
                root1 = tk.Tk()
                mess = "Sorry but your video does not satisfy the constraints such as proper length or proper fps"
                tk.messagebox.showwarning('Error',mess)
                root1.destroy()

    def startRec(self):
        cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'XVID') 
        self.input_name = "inp"+str(time.strftime("%Y%m%d-%H%M%S"))+".avi"
        temp = str(os.path.dirname(os.path.abspath(__file__)))
        self.output_name = temp+"\out"+str(time.strftime("%Y%m%d-%H%M%S"))+".avi"
        out = cv2.VideoWriter(self.input_name, fourcc, 8.0, (640, 480)) 
        flag = FALSE
        while(flag==False):
            ret,frame = cap.read()
            out.write(frame)
            cv2.imshow('video',frame)
            if cv2.waitKey(1) & 0xFF == ord('a'): 
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        # y1,y2 = self.startpro()
        # speed2.functionCall('output11.avi','finaloutput111.avi',y1,y2)

    def open_file(self):

        self.pause = False

        self.filename = filedialog.askopenfilename(title="Select file", filetypes=(("MP4 files", "*.mp4"),("M4V files","*.m4v"),
                                                                                         ("WMV files", "*.wmv"), ("AVI files", "*.avi")))
        # print(self.filename)
        self.input_name = self.filename
        temp = str(os.path.dirname(os.path.abspath(__file__)))
        self.output_name = temp+"\out"+str(time.strftime("%Y%m%d-%H%M%S"))+".avi"
        # y1,y2 = self.startpro()
        # speed2.functionCall(self.filename,'file_output.avi',y1,y2)
        # Open the video file
        # self.cap = cv2.VideoCapture(self.filename)

        # self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        # self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # ratio = self.width/self.height
        # if self.width > 720 or self.height > 720:
        #     self.width = int(720)
        #     self.height = int(self.width/ratio)
        # self.canvas.config(width = self.width, height = self.height)


    # def get_frame(self):   # get only one frame

    #     try:

    #         if self.cap.isOpened():
    #             ret, frame = self.cap.read()
    #             return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    #     except:
    #         messagebox.showerror(title='Video file not found', message='Please select a video file.')


    # def play_video(self):

    #     # Get a frame from the video source, and go to the next frame automatically
    #     ret, frame = self.get_frame()

    #     if ret:
    #         # frmae = frame.resize((self.height,self.width))
    #         temp = PIL.Image.fromarray(frame)
    #         temp = temp.resize((self.width,self.height))
    #         self.photo = PIL.ImageTk.PhotoImage(image = temp)
    #         self.canvas.create_image(0, 0, image = self.photo, anchor = NW)

    #     if not self.pause:
    #         self.window.after(self.delay, self.play_video)

    # Release the video source when the object is destroyed
    # def __del__(self):
    #     if self.cap.isOpened():
    #         self.cap.release()

##### End Class #####


# Create a window and pass it to videoGUI Class
while 1:
    root = tk.Tk()
    root.protocol("WM_DELETE_WINDOW",on_close)
    videoGUI(root, "Human Speed Detection")