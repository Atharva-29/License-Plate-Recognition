from tkinter import *
import tkinter
from tkinter import filedialog
import predln

def pred_number():

   provided_image = filedialog.askopenfilename(initialdir = "/",title = "Select image")
   number= predln.ret_pred(provided_image)
   res.config(text="Predicted License Number is : %s" %number)

root = Tk()
root.geometry('1440x900+200+100')
root.title("Recognition")

frame = Frame(root, relief=RIDGE, borderwidth=2)
frame.pack(fill=BOTH,expand=1)
frame.configure(background='white')

label = Label(frame, text="License Number Recognition", bg='white', font=('Arial 35 bold'))
label.pack(side=TOP)

background_label = Label(frame)
background_label.pack(side=TOP)
   
feed_image=Button(frame, padx=5, pady=5, width=35, bg='white', fg='black', relief=RAISED, command = pred_number, text='Upload Image', font=('helvetica 15 bold'))
feed_image.place(x=550,y=200)

res=Label(frame)
res.place(x=550,y=250)

root.mainloop()
