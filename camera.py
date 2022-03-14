import cv2
# 创建一个VideoCapture对象, 会调取摄像头
cap = cv2.VideoCapture(0)
while True:
    # 逐帧捕获
    ret, frame = cap.read()
    # ret 为布尔值, 代表有没有读取到图片, frame 为截取到的一帧率=的图片
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("frame", gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#结束循环释放VideoCapture对象
cap.release()
cv2.destroyAllWindows()

