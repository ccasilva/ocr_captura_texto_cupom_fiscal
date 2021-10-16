import pytesseract
import cv2
import numpy as np
import os

per = 25
pixelThreshold = 500

roi = [[(278, 666), (1714, 754), 'text', 'doc']]

# roi = [[(596, 218), (1206, 306), 'text', 'nomeLoja'],
#        [(596, 308), (1158, 378), 'text', 'cnpj'],
#        [(540, 908), (1386, 976), 'text', 'produto1'],
#        [(1528, 978), (1742, 1056), 'text', 'valor1'],
#        [(428, 1058), (1390, 1126), 'text', 'produto2'],
#        [(1520, 1132), (1738, 1212), 'text', 'valor2'],
#        [(424, 1214), (1110, 1288), 'text', 'produto3'],
#        [(1566, 1290), (1742, 1364), 'text', 'valor3'],
#        [(1580, 1472), (1790, 1548), 'text', 'valorTot']]


imgQ = cv2.imread("cut_2_sem_qr_code.png")
h,w,c = imgQ.shape

orb = cv2.ORB_create(1000)
kp1, des1 = orb.detectAndCompute(imgQ,None)

path = 'src'
myPicList = os.listdir(path)

for j,y in enumerate(myPicList):
    img = cv2.imread(path +"/"+y)
    kp2, des2 = orb.detectAndCompute(img,None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2,des1)
    matches.sort(key= lambda x: x.distance)
    good = matches[:int(len(matches)*(per/100))]
    imgMatch = cv2.drawMatches(img,kp2,imgQ,kp1,good[:100],None,flags=2)

    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    M, _ = cv2.findHomography(srcPoints,dstPoints,cv2.RANSAC,5.0)
    imgScan = cv2.warpPerspective(img,M,(w,h))

    imgShow = imgScan.copy()
    imgMask = np.zeros_like(imgShow)

    myData = []

    print(f'################### Extraindo dados do formulário {j} ###################')

    for x,r in enumerate(roi):
        cv2.rectangle(imgMask, (r[0][0],r[0][1]),(r[1][0],r[1][1]),(0,255,0),cv2.FILLED)
        imgShow = cv2.addWeighted(imgShow,0.99,imgMask,0.1,0)

        imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        # captura os dados e apresenta eles
        cv2.imshow(str(x), imgCrop)

        if r[2] == 'text':
            print(f'{r[3]} : {pytesseract.image_to_string(imgCrop)}')
            myData.append(pytesseract.image_to_string(imgCrop))
        if r[2] == 'box':
            imgGray = cv2.cvtColor(imgCrop,cv2.COLOR_BGR2GRAY)
            imgThresh = cv2.threshold(imgGray,170,255,cv2.THRESH_BINARY_INV)[1]
            totalPixels = cv2.countNonZero(imgThresh)
            if totalPixels > pixelThreshold: totalPixels = 1
            else: totalPixels = 0
            print(f'{r[3]} : {totalPixels}')
            myData.append(totalPixels)

        cv2.putText(imgShow,str(myData[x]),(r[0][0], r[0][1]),
                    cv2.FONT_HERSHEY_PLAIN,2.5,(0,0,255),4)

    # with open('DataOutput.csv', 'a+') as f:
    #
    #     for data in myData:
    #         f.write((str(data)+';'))
    #     f.write('\n')


    imgShow = cv2.resize(imgShow,(w//3,h//3))
    print(myData)
    cv2.imshow(y+"2",imgShow)

cv2.waitKey(0)