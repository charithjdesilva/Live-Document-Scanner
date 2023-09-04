import cv2
import numpy as np
from image_stacking import stackImages

imgWidth = 640
imgHeight = 480


# videoCap = cv2.VideoCapture(2)
videoCap = cv2.VideoCapture("https://192.168.8.100:4343/video") # chnage accoring to the ip and port of the cam
videoCap.set(3, imgWidth)
videoCap.set(4, imgHeight)
videoCap.set(10, 150)
# videoCap.set(cv2.CAP_PROP_BRIGHTNESS , 150)   

# preprocess the image to detect edges properly
def preprocessImage(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # convert to gray image
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)    # apply a little bit of blur
    imgCanny = cv2.Canny(imgBlur, 150, 150)    # apply canny edge detector

    # some times if there is shadows it might not take the edges properly, to solve that we dialte and erode the image
    kernel = np.ones((5,5))

    imgDialate = cv2.dilate(imgCanny, kernel, iterations=2)     # increase the thickness of the edges
    imgThres = cv2.erode(imgDialate, kernel, iterations=1)     # decrease the thickness of the edges

    return imgThres

# getContours of the image, in here we are finding the largest one
def getContours(imgCropped):
    biggest = np.array([[(0, 0)], [(0, 0)], [(0, 0)], [(0, 0)]])
    maxArea = 0

    contours, hierarchy = cv2.findContours(imgCropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # find contours of an image  (RETR_EXTERNEL is used to find outer corners)

    # loop through each contour
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # print(area)

        # give minimum width for the area, so that it does not detect any noise
        if area > 5000:
            # draw contours to see them clearly
            cv2.drawContours(imgContour, cnt, -1, (255,0,0),3)  # will be drawn in the image provided
            
            # curve length helps to approximate the corners of the shapes
            peri = cv2.arcLength(cnt, True) #length of the each contour, True means closed

            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)    # approximate how many corner points, , True means shape is closed
            # print(approx)   # will print the coordinates of corner points
            # print(len(approx))  # number of corner points
    
            # create object corners
            objCor = len(approx)    # length should be 4, paper has 4 corners

            # will give us the biggest area with 4 corners
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv2.drawContours(imgContourMax, biggest, -1, (255,0,0),20)  # biggest contour will be drawn in the image provided
    print(maxArea)
    return biggest

def reOrderPoints(contourPoints):
    # print(contourPoints.shape)
    contourPoints = contourPoints.reshape((4,2))    # remove the redundant 1 in contour shape (4= no. of corners, 1 , 2=xy)

    # output shape should be same as what we are recieving (4,1,2)
    contourPointsNew = np.zeros((4,1,2),np.int32)

    add = contourPoints.sum(1)      # width + height will be the max, starting pt + end pt will be the min
    # print('add',add)

    contourPointsNew[0] = contourPoints[np.argmin(add)] #np.argmin(add) will find the index of the smallest value
    contourPointsNew[3] = contourPoints[np.argmax(add)]
    # print("New points", contourPointsNew)

    # finding the order of the middle 2 points
    # check the differnce of them, if it provides positive value points are in order, else swap them
    diff = np.diff(contourPoints, axis=1)
    contourPointsNew[1] = contourPoints[np.argmin(diff)]
    contourPointsNew[2] = contourPoints[np.argmax(diff)]
    # print("Ordered points",contourPointsNew)
    return contourPointsNew

# get warp perspective of the biggest contour
def getWarpPerspective(img,biggest):
    biggest = reOrderPoints(biggest)
    pts1 = np.float32(biggest)      # our contour points should be in order according to pts2
    pts2 = np.float32([[0,0],[imgWidth,0],[0,imgHeight],[imgWidth,imgHeight]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)    # get transofrmation matrix required for the perspective
    imgOutput = cv2.warpPerspective(img,matrix, (imgWidth, imgHeight))      # get the image transformed into the warp perspective

    return imgOutput

while True:
    success, img = videoCap.read()
    img = cv2.resize(img,(imgWidth, imgHeight)) # resize the image
    imgCrop = img[10:480, 0:640]
    imgContour = imgCrop.copy()
    imgContourMax = imgCrop.copy()

    imgThres = preprocessImage(imgCrop)     # preprocess the image to detect edges
    biggest = getContours(imgThres)   # get the biggest contour
    # get the warp perspective of the contour image
    print(biggest)
    imgWarped = getWarpPerspective(imgCrop,biggest)
    # Transpose the image , to potrait the image
    imgTranspose = cv2.transpose(imgWarped)
    # Flip the image horizontally
    imgFlipped = cv2.flip(imgTranspose, 1)
    imgCropFlipped = imgFlipped[15:imgFlipped.shape[0]-15, 15:imgFlipped.shape[1]-15]   #remove 15px from each side
    imgCropFlipped = cv2.resize(imgCropFlipped, (imgWidth, imgHeight))

    # print(imgContour.shape)

    imgResult = stackImages((0.6),[[imgCrop,imgThres],[imgContour,imgWarped]])

    # imgHorStack = stackImages(1,[imgResult, imgFlipped])
    # imgHorStack = np.hstack((imgResult, imgFlipped))

    print(imgResult.shape)
    print(imgFlipped.shape)

    cv2.imshow("Result", imgResult)
    cv2.imshow("Result Warped", imgFlipped)

    # cv2.imshow("ResultStacked", imgHorStack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
