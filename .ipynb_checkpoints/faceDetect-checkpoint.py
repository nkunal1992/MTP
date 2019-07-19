# OpenCV program to detect face in real time
# import libraries of python OpenCV 
# where its functionality resides

import cv2 


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') 

#lecture video name
vid='iitm.mp4'


# capture frames from a file
cap = cv2.VideoCapture(vid)

# Python List declaration
t1 = []
f1 = []
x1 = []
x2 = []



# loop runs if capturing has been initialized.
while 1: 
 
    # reads frames from a camera
    ret, img = cap.read()

    # Reached last frame ? Then break from infinite loop
    if ret == False:
        break

    # convert to gray scale of each frames
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create the list of all frames
    f1.append(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
    
   
    # Detects faces of different sizes in the input image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
 
    for (x,y,w,h) in faces:
        # To draw a rectangle in a face 
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2) 
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        x = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        t1.append(x)
        print 'Face Detected at Frame no.', x
        cv2.putText(img,'Face Detected',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
 
        # Detects eyes of different sizes in the input image
        #eyes = eye_cascade.detectMultiScale(roi_gray) 
 
         
    # Display an image in a window
    cv2.imshow('img',img)
 
    # Wait for Esc key to stop
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        x1 = list(set(f1) - set(t1))
        break

# compute the difference to find the frames containing only slides
#x1 = list(set(f1) - set(t1))
print 'This section reached'


# Close the window
cap.release()
x1 = list(set(f1) - set(t1))
print x1


cap = cv2.VideoCapture(vid)

while True:
    ret,img = cap.read()
    
    # Reached last frame ? Then break from infinite loop
    if ret == False:
        break

    #Locate the bounding box across various titles
    roi_1 = cv2.rectangle(img,(52,8),(350,27),(0,255,0),2)
    roi_2 = cv2.rectangle(img,(45,35),(350,85),(0,0,255),2)
    

    # convert to gray scale of each frames
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Retrieve the current frame number
    x = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    for i in x1:
        if x == i:
            cv2.imshow('Slides Only',img)
 
    # Wait for Esc key to stop
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
        


cap.release()


# De-allocate any associated memory usage
cv2.destroyAllWindows() 
