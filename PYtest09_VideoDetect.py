
# 捕获摄像头图像
# 对视频流进行运动目标检测

import cv2 as cv
import numpy as np
import time

# 0本机摄像头
v = cv.VideoCapture(0)
time.sleep(1)

# 初始化，表示背景尚未形成
bg = None
# 膨胀参数
es = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9,4))

while( v.isOpened() ):
    ret, frame0 = v.read()
    if ret == True :
        cv.imshow( 'Orignal', frame0 )

        frame = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)

        # 高斯平滑处理
        frame = cv.GaussianBlur(frame, (21,21), 0)

        # 如果是第一帧，跳过去
        if bg is None :
            bg = frame.copy().astype("float")
            continue
        
        # 让背景持续累积修正
        cv.accumulateWeighted(frame, bg, 0.05) #最后一个参数，越小表示背景变化越缓慢
        cv.imshow( 'Backgroud', cv.convertScaleAbs(bg) )

        # 当前帧跟背景进行比较
        diff = cv.absdiff(cv.convertScaleAbs(bg), frame)
        cv.imshow( 'absDiff', diff )
        # 二值化
        diff = cv.threshold(diff, 25, 255,cv.THRESH_BINARY)[1]
        # 膨胀
        diff = cv.dilate(diff, es, iterations=2)

        # 在图像中找轮廓
        contours, _ = cv.findContours( diff.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        frame1 = frame0.copy()
        frame1[frame1>0] = 0
        for c in contours :
            # 太小的不要
            if cv.contourArea(c) < 1500 :
                continue
            (x,y,w,h) = cv.boundingRect(c)
            cv.rectangle(frame0, (x,y), (x+w,y+h), (0,255,0), 2)
            # 把轮廓中的图像部分copy过去
            # frame1[x:x+w, y:y+h] = frame0[x:x+w, y:y+h]
            frame1[y:y+h, x:x+w] = frame0[y:y+h, x:x+w]

        cv.imshow( 'Target', frame0 ) # 在原图上用矩形画出move部分
        cv.imshow('Only Move', frame1) # 只显示move部分
        cv.imshow( 'Diss', diff ) # 图像和背景的差异部分
        
        k = cv.waitKey(50)
        if k == ord('q'):
            break

v.release()
cv.waitKey(0)
cv.destroyAllWindows()
