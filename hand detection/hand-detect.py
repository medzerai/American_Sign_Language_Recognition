import cv2
from cvzone.HandTrackingModule import HandDetector
from PIL import Image as im
import keyboard
import numpy as np
from keras.models import load_model

# preparer la capture video du camera 
cap = cv2.VideoCapture(0)
key = cv2.waitKey(1)

# initialiser le detecteur des main
detector = HandDetector(detectionCon=0.8,maxHands=2)

# predire a partir de l'image du main donnée
def predictImage(image):
    train=np.array(image,dtype="float32")
    train = train[:, 1:] /255.0
    train = cv2.resize(train, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
    train=train.reshape(1,28,28,1)
    
    class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y' ]
    
    model1 = load_model('../saved_models/model1.hdf5')
    model2 = load_model('../saved_models/model2.hdf5')
    model3 = load_model('../saved_models/model3.hdf5')
    models = [model1, model2, model3]
    
    preds = [model.predict(train) for model in models]
    preds=np.array(preds)
    summed = np.sum(preds, axis=0)
    res = np.argmax(summed, axis=1)
    return class_names[res[0]]

# préparer l'image en donnant la valeur d'espacement (o), la table de trame principale (bbox) et la matrice de l'image capturée(imCopy)
def prepImage(o,bbox,imCopy):
    #bbox=[X,Y,width,height]
    xb,yb,wb,hb=bbox
    xb=xb-o
    yb=yb-o
    if wb<hb:
        wb=int(wb+((hb-wb)/2))
        xb=int(xb-((hb-wb)/2))
    elif wb>hb:
        hb=int(hb+((wb-hb)/2))
        yb=int(yb-((wb-hb)/2))
                    
    if xb<0:
        xb=0
    else:
        if xb+wb+o>dim[1]:
            wb=dim[1]-xb              
    if yb<0:
        yb=0
    else:
        if yb+hb+o>dim[0]:
            hb=dim[0]-yb            
    b1=[xb,yb,wb,hb]         
    imk = imCopy[b1[1]:b1[1]+b1[3]+o, b1[0]:b1[0]+b1[2]+o]
    gray=cv2.cvtColor(imk, cv2.COLOR_BGR2GRAY)                    
            
    img_gray=gray
    h,w=gray.shape[0],gray.shape[1]
    if h>w:
        img_gray=np.pad(gray, [(0, ),(int((h-w)/2), )], mode='constant', constant_values=255)
    elif w>h:
        img_gray=np.pad(gray, [(int((w-h)/2), ),(0, )], mode='constant', constant_values=255)
    return img_gray

# cc c'est la variable qu'on va l'utiliser pour optimer la prediction des plusieurs images dans une seconde
cc=0
# ok est la variable d'activation et de désactivation de la prédiction
ok=False
# default value de la prediction
non="?"
while True:
    
    # capture video
    success,img=cap.read()
    dim=img.shape
    # faire une copie de l'image pour la prediction 
    imCopy=img.copy()
    #detecter les mains
    hands,img=detector.findHands(img)
    #hand -dict {lmList -bbox -center -type}
    
    # cliquer sur "o" pour activer prediction mode
    if keyboard.is_pressed('o'): 
        ok=not ok
        print(ok)
    
    if hands:
        hand1=hands[0]
        lmList1=hand1["lmList"] # list of 21 Landmarks points for each finger..
        bbox1=hand1["bbox"] # bounding box x,y,w,h
        centerPoint1= hand1["center"] #center of the hand x,y
        handType1=hand1["type"]
        
        #afficher un rectangle bleu pour afficher les predictions à la detection des mains
        cv2.rectangle(img, (20,20), (150,200), (255,0,0),cv2.FILLED)
        cv2.putText(img, non, (40,150), cv2.FONT_HERSHEY_PLAIN, 7, (255,255,255),5)

        # predire toujours les signes des la main droite
        if(handType1=="Right"):
            if cc>=20:
                cc=0
                if ok :
                    # afficher la prediction
                    cv2.rectangle(img, (20,20), (150,200), (255,0,0),cv2.FILLED)
                    X=prepImage(20,bbox1,imCopy)
                    non=str(predictImage(X))
                    cv2.putText(img, non, (40,150), cv2.FONT_HERSHEY_PLAIN, 7, (255,255,255),5)
            else:
                cc=cc+1
        
        # le meme cas de prediction main à la detection de deux mains
        if(len(hands)==2):
            hand2=hands[1]
            lmList2=hand2["lmList"] # list of 21 Landmarks points for each finger
            bbox2=hand2["bbox"] # bounding box x,y,w,h
            centerPoint2= hand2["center"] #center of the hand x,y
            handType2=hand2["type"]
            
            if(handType2=="Right"):
                if cc>=20:
                    cc=0
                    if ok :
                        cv2.rectangle(img, (20,20), (150,200), (255,0,0),cv2.FILLED)
                        X=prepImage(20,bbox2,imCopy)
                        non=str(predictImage(X))
                        cv2.putText(img, non, (40,150), cv2.FONT_HERSHEY_PLAIN, 7, (255,255,255),5)
                else:
                    cc=cc+1
            
        # cliquer sur "s" pour sauvegarder l'image du main detecté
        if keyboard.is_pressed('s'): 
            img_gray=prepImage(20,bbox1,imCopy)
            data = im.fromarray(img_gray)
            data.save('./dataHands/opencv'+str(np.random.randint(100))+'.png')
            print("image saved")
            
        
    #afficher l'image du video
    cv2.imshow("Image",img)
    cv2.waitKey(1)