# import libraries

from gtts import gTTS
import pygame,picamera
import pytesseract
import os,time
from PIL import Image
import numpy as np
import cv2

import datetime
import pyaudio
from subprocess import call
import speech_recognition as sr
import shelve,os

#to get foldername for captured image by date 

dt=datetime.datetime.now()
month=dt.month
day=dt.day
folderName=str(month)+'-'+str(day)

#create the folders if its new day

if folderName not in os.listdir('/home/pi/project'):
    validNewFolder=True
    currentDir = '/home/pi/project/'+str(folderName)
    os.makedirs(currentDir)
    os.makedirs(currentDir+'/input') #folder for captured images
    os.makedirs(currentDir+'/audio') #folder for audio outputs
    os.makedirs(currentDir+'/process') #folder for images after image processing
    os.makedirs(currentDir+'/textfile')#folder for textfile to write output text
    os.makedirs(currentDir+'/pdffile')#folder for pdffiles to write output text
    imageNum=0 # for naming of each image by counter
    shelfFile = shelve.open('number') #to store the count in shelve
    shelfFile['imageNum']=imageNum
    shelfFile.close()
else:
    #if folder is already there(not a new day)
    shelfFile = shelve.open('number')
    imageNum = shelfFile['imageNum']
    shelfFile.close()
        
def recognition(): #function for voice recognition part
    
    r = sr.Recognizer() # voice for assistant
    r.energy_threshold=4000
    os.system("omxplayer /home/pi/Downloads/plucky.mp3")
    with sr.Microphone() as source:
        
        print ('listening..')
        time.sleep(1)
        audio = r.listen(source) #listen to blind people's command
        print ('processing')
        #voice recognition part
        try:
            message = (r.recognize_google(audio, language = 'en-us', show_all=False))
            #call(["espeak", message])
            print(message)
            
        except sr.UnknownValueError:
            print("Could not understand audio")
            #time.sleep(3)
            recognition()
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
            #time.sleep(3)
            recognition()
 
    
    return message #return text recognized by voice recognition

def picam(imageNum):
    # fuction for capture image of page

    os.chdir('/home/pi/project/'+str(folderName)+'/input')#change directory to input folder
    image=str(imageNum)+'.jpg' #name for image
    camera=picamera.PiCamera()# initialize picamera
    camera.capture(image) #capture image
    camera.close() #close picamera
    
def image_process(imageNum):
    #function for image process
    
    os.chdir('/home/pi/project/'+str(folderName)+'/input') # change the directory to folder
    image=str(imageNum)+'.jpg'
    image = cv2.imread(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #cv2.imshow('bin',thresh)
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h,w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    kernal = np.ones((1,1),np.uint8)
    img = cv2.dilate(rotated, kernal,iterations=1)
    img = cv2.erode(rotated, kernal, iterations=1)
    

    #img = cv2.adaptiveThreshold(img,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31,2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
     
    # threshold the image, setting all foreground pixels to
    # 255 and all background pixels to 0
    thresh = cv2.threshold(gray, 0, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    os.chdir('/home/pi/project/'+str(folderName)+'/process')
    
    cv2.imwrite(str(imageNum)+'.png',thresh)
    #def ocr
    
    im=Image.open(str(imageNum)+'.png')
    text=pytesseract.image_to_string(im, lang='eng')
    return text #return the text extracted from image
def tts(text,imageNum):
    #function for audio conversion
    
    os.chdir('/home/pi/project/'+str(folderName)+'/audio')
    name= str(imageNum)+'.mp3'
    tts=gTTS(text=text, lang='en')
    tts.save(name)

    pygame.mixer.init()
    pygame.mixer.music.load(name)
    pygame.mixer.music.play()

def musicPlay(imageNum):
    #function for play the audio
    
    os.chdir('/home/pi/project/'+str(folderName)+'/audio')
    name= str(imageNum)+'.mp3'
    pygame.mixer.init()
    pygame.mixer.music.load(name)
    pygame.mixer.music.play()
    
def textfile(text,imageNum):
    #function for textfile, pdf conversion 
    
    textPath= os.chdir('/home/pi/project/'+str(folderName)+'/textfile')
    filename=str(imageNum)+'.txt'
    textfile=open(filename, 'w')
    textfile.write(text)
    textfile.close()
    textPath=str(textPath) + filename
    pdfPath='/home/pi/project/'+str(folderName)+'/pdffile/'+str(imageNum)+'.pdf'
    path=pdfPath+' '+textPath
    cmnd='python3 txt2pdf.py -o '+path
    os.system(cmnd)
    
def reader(imageNum):
    #function for structure the program
    
    picam(imageNum)
    #text = image_process(imageNum)
    text = 'good morning'
    textfile(text, imageNum)
    os.system('''flite -voice slt -t "place the next page and say ok
              for better performence"''')
    chumm=recognition()
    if 'ok' in chumm:
        import threading
        imageNum2=imageNum
        imageNum+=1
        threadObj = threading.Thread(target=main, args=['start reading',imageNum])
        threadObj.start()
        tts(text,imageNum2)   #thrading
    else:
        tts(text,imageNum)
        imageNum+=1
    
    
def main(message,imageNum):
    #main program
    
    if 'reading' in message:
        os.system('flite -voice slt -t "place the book or page on table and say ok"')
        #time.sleep(1)
        
        message=recognition()
        if 'ok' in message:
            
            reader(imageNum)
            shelfFile = shelve.open('number')
            shelfFile['imageNum']=imageNum
            shelfFile.close()
            os.system('flite -voice slt -t "do you want to hear once more?"')
            message = recognition()
            if 'yes' in message:
                musicPlay(imageNum-1)
            #elif 'no' in message:
            #break
    if ('previous' or 'page') in message:
        shelfFile = shelve.open('number')
        imageNum=shelfFile['imageNum']
        shelfFile.close()
        imageNum=imageNum-2
        musicPlay(imageNum)
    if 'shutdown' in message:
        os.system("sudo shutdown")
    if ('refresh' or 'reboot') in message:
        os.system("sudo reboot")
    if ('pdf' or 'convert') in message:
        import PyPDF2
        os.system('flite -voice slt -t "which date of pages do you want to convert?"')
        message=recognition()
        if 'today' in message:
            os.chdir('/home/pi/project/'+ folderName+'/pdffile')
            
        if 'yesterday' in message:
            dt=datetime.datetime.now()
            month=dt.month
            day=dt.day
            day-=1
            date=str(month)+'-'+str(day)
            os.chdir('/home/pi/project/'+date+'/pdffile')
        else:
            dt=datetime.datetime.strptime(str(message),'%B %d')
            month=dt.month
            day=dt.day
            date=str(month)+'-'+str(day)
            os.chdir('/home/pi/project/'+date+'/pdffile')
        
                 
        
            
            
                

message=recognition()
main(message,imageNum)
