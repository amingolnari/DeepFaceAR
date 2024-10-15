import warnings
warnings.filterwarnings("ignore", category = FutureWarning)
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import time
from keras.models import load_model
import _pickle as pickle

def main():
    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)
    else:
        print('OpenCL not supported - Work on CPU')
    numClasses = 100
    try:
        model = load_model('DeepFaceModel.h5')
    except OSError as err:
        print('Could not find or open Deep Model ...!')
        print("Exception Error: \n {0}".format(err))
        return
    if model._built:
        print('Deep model was loaded')
    try:
        Labels = GetLabels('labels.txt')
    except OSError as err:
        print('Could not find or open labels file ...!')
        print("Exception Error: \n {0}".format(err))
        return
    if len(Labels) < numClasses:
        print('Face labels not enough ...!')
        return
    else:
        print('Face labels was loaded')
    try:
        detector = cv2.xfeatures2d.SURF_create(400)
        Target, KP, Des = GetTargetsAndKP()
    except OSError as err:
        print('Could not find or open AR-Targets ...!')
        print("Exception Error: \n {0}".format(err))
        return
    if len(Target) == numClasses:
        print('AR-Target images loaded')
    else:
        print('AR-Targets not enough ...!')
        return
    faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    if not faceDetector.empty():
        print('Face detector was loaded')
    else:
        print('Could not find or open face detector .xml file ...!')
        return
    H, W, C = 100, 100, 3
    Cap = cv2.VideoCapture(0)
    ret, frame = Cap.read()
    if ret == False:
        print('Could not open video or webcam !!')
        Cap.release()
        return
    cv2.namedWindow('DeepAR Face-Keras', 0)
    cv2.resizeWindow('DeepAR Face-Keras', frame.shape[1], frame.shape[0])
    Kcheck = 11
    preds = np.full([Kcheck, 1], -1, dtype = 'int')
    a = maxrep = err = 0
    processing = False
    BFm = cv2.BFMatcher(cv2.NORM_L1, crossCheck = False)
    while (Cap.isOpened):
        ret, frame = Cap.read()
        if ret == False:
            break
        frame = cv2.resize(frame, ((int)(frame.shape[1]/2), (int)(frame.shape[0]/2)))
        faces = Classifier(frame, faceDetector)
        startTime = time.clock()
        for (x, y, w, h) in faces:
            if w < 100.0:
                break
            imageROI = cv2.resize(frame[y : y+h, x : x+w], (H, W))
            InputData = np.array(imageROI, dtype = 'float32') / 255.0
            InputData = InputData.reshape([1, H, W, C])
            pred = model.predict(InputData, verbose = 1)
            sortPreds = sorted([(p, i) for i, p in enumerate(pred[0])], reverse = True)
            if sortPreds[0][0]*100 < 9.9:
                break
            if (sortPreds[0][1] == 99):
                preds[a] = sortPreds[0][1]-97
                a += 1
            elif (sortPreds[0][1] >= 0) and (sortPreds[0][1] < 99):
                preds[a] = sortPreds[0][1]+1
                a += 1
            if a > len(preds)-1 or not np.any(preds == -1):
                processing = True
                a = 0
                maxrep = MaxRep(preds)
            break
        endTime = time.clock()
        if processing == True:
            kp, des = detector.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
            matches = BFm.knnMatch(des, Des[maxrep], k = 2)
            goodPoints = GoodPoints(matches)
            if len(goodPoints) > 14:
               pts1 = np.float32([kp[m.queryIdx].pt for m in goodPoints]).reshape(-1, 1, 2)
               pts2 = np.float32([KP[maxrep][m.trainIdx].pt for m in goodPoints]).reshape(-1, 1, 2)
               frame = Homography(frame, Target[maxrep], pts1, pts2, frame.shape[0], frame.shape[1])
               err = 0
            else:
                cv2.putText(frame, 'Can not extract 3D coordinates',
                            (20, 20), cv2.FONT_ITALIC, .6, (20, 100, 255), 2)
                err += 1
                if err > 3:
                    processing = False
                    a = 0
                    preds = np.full([Kcheck, 1], -1, dtype = 'int')

        if endTime - startTime < 0.00005:
            processing = False
            a = 0
            preds = np.full([Kcheck, 1], -1, dtype = 'int')
            cv2.putText(frame, 'Please Wait For Processing ...',
                        (20, 20), cv2.FONT_ITALIC, .6, (255, 0, 0), 2)
        cv2.imshow('DeepAR Face-Keras', frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
    Cap.release()
    return

def Homography(frame, Target, pts1, pts2, H, W):
    M, status = cv2.findHomography(pts2, pts1, cv2.RANSAC)
    if status.any(0)[0]:
        dst = cv2.warpPerspective(Target, M, (W, H))
        _, mask = cv2.threshold(cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY), 10, 1, cv2.THRESH_BINARY_INV)
        mask = cv2.erode(mask,(3,3))
        mask = cv2.dilate(mask,(3,3))
        for c in range(0,3):
            frame[:, :, c] = dst[:,:,c] * (1 - mask[:,:]) + frame[:,:,c] * mask[:,:]
    else:
        print('Homography Matrix is None')
    return frame

def Probability(Preds, Probs, ClassIdx):
    prob = 0.0
    cunt = 0
    for i in range(len(Preds)):
        if Preds[i] == ClassIdx:
            prob += Probs[i]
            cunt += 1
    return (float)(prob[0]/cunt)

def GoodPoints(matches):
    points = []
    for m, n in matches:
        if m.distance < 0.6*n.distance:
            points.append(m)
    return points

def MaxRep(Array):
    uniq, cunt = np.unique(Array, return_counts = True)
    maxrep = uniq[np.argmax(np.asarray(cunt))]
    return maxrep

def Classifier(image, faceDetector):
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(grayImage, 1.1, 5)
    return faces

def GetLabels(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content

def GetTargetsAndKP():
    KP = []
    Des = []
    KData = pickle.load(open("KData.p", "rb"))
    for k in range(100):
        temKey, temDes = ReadKeyData(KData[k])
        KP.append(temKey)
        Des.append(temDes)
    Target = np.load('ARTargets.npy')
    return Target, KP, Des

def ReadKeyData(array):
    key = []
    des = []
    for point in array:
        feature = cv2.KeyPoint(x = point[0][0], y = point[0][1],
                               _size = point[1], _angle = point[2],
                               _response = point[3], _octave = point[4],
                               _class_id = point[5])
        descriptor = point[6]
        key.append(feature)
        des.append(descriptor)
    return key, np.array(des)

if __name__=='__main__':
    main()