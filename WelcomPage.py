import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image
from second_frame import SecondFrame
from third_frame import ThirdFrame
import cv2

# create a window
window = tk.Tk()

# set the title
window.title("Welcome Page")

# set the size
window.geometry("800x600")
# window.attributes('-fullscreen', True)


# load the video
cap = cv2.VideoCapture('TradeVideo.mp4')

# create a canvas to display the video
canvas = tk.Canvas(window, width=800, height=600)
canvas.pack()

# define a function to update the video on the canvas

def show_second_page():
    canvas.pack_forget()
    second_page = SecondFrame(window, master=window)
    second_page.pack(fill="both", expand=True)
def show_LinearReg():
    canvas.pack_forget()
    second_page = ThirdFrame(window, master=window)
    second_page.pack(fill="both", expand=True)

def update():
    ret, frame = cap.read()
    if ret:
        img = Image.fromarray(frame)
        img = img.resize((800, 600))
        img_tk = ImageTk.PhotoImage(image=img)
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        canvas.img_tk = img_tk
    window.after(30, update)

# call the update function to start playing the video
update()

# create a style for the buttons
style = ttk.Style()

# configure the style for the buttons
style.configure('TButton', font=('Arial', 16), foreground='black', background='#f30b26', width=10, padding=10)

# create buttons
button1 = ttk.Button(window, text="LSTM",width=20)
button2 = ttk.Button(window, text="Linear Regression",command=show_LinearReg,width=20)
button3 = ttk.Button(window, text="Lasso",width=20)
button4 = ttk.Button(window, text="Ridge",width=20)
button5 = ttk.Button(window, text="Sentiment Analysis", command=show_second_page,width=20)
# place the buttons on the window
button1.place(x=400, y=200, anchor=tk.CENTER)
button2.place(x=400, y=270, anchor=tk.CENTER)
button3.place(x=400, y=340, anchor=tk.CENTER)
button4.place(x=400, y=410, anchor=tk.CENTER)
button5.place(x=400, y=480, anchor=tk.CENTER)

# start the GUI
window.mainloop()
