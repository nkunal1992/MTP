import cv2

img = cv2.imread('Narayan.jpg')

while True:
    
    cv2.imshow('Me',img)
    
    #if i've waited atleast 2 ms and i've pressed Esc
    if cv2.waitKey(2) & 0xFF == 27:
        break
        
cv2.destroyAllWindows()