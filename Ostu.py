'''
Phương pháp Otsu
    Đây là phương pháp xác định ngưỡng cho toàn bộ ảnh. Phương pháp này sẽ tìm một ngưỡng để
    phân chia các điểm ảnh vào hai lớp tiền cảnh (đối tượng) và nền. Giá trị ngưỡng được xác định
    sao cho "khoảng cách" giữa các điểm ảnh trong mỗi lớp là nhỏ nhất, điều này tương ứng với
    khoảng cách giữa hai lớp là lớn nhất. Việc phân chia này dựa trên các giá trị trong histogram
    của ảnh. Các bước để xác định ngưỡng Otsu của ảnh được tiến hành như sau
        1. Tính histogram của ảnh: {pi}
        2. Tính lũy tích cho nền Pb và tiền cảnh Pf
        3. Tính độ lệch chuẩn của nền sigma_b và tiền cảnh sigma_f
        4. Các định hàm khoảng cách: var(between class) hoặc var(within class)
        5. Ngưỡng t(otsu) là đối số để hàm var(between class) lớn nhất hoặc hàm var(within class) đạt giá trị nhỏ nhất
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('Photos/park.jpg',0)
blur = cv2.GaussianBlur(img,(5,5),0) #Filter

# find normalized_histogram, and its cumulative distribution function
hist = cv2.calcHist([blur],[0],None,[256],[0,256])
#print(hist)
#print(hist.max())
hist_norm = hist.ravel()/hist.max()
#print(hist_norm)
Q = hist_norm.cumsum() # Trả v tổng tích lũy của các phần tử mảng dọc theo trục đã cho
print(Q)
bins = np.arange(256)

fn_min = np.inf
print(fn_min)
thresh = -1
for i in range(1,256):
    p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
    q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
    b1,b2 = np.hsplit(bins,[i]) # weights

    # finding means and variances
    m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
    v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2

    # calculates the minimization function
    fn = v1*q1 + v2*q2
    if fn < fn_min:
        fn_min = fn
        thresh = i

# find otsu's threshold value with OpenCV function
ret, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print (thresh,ret)


