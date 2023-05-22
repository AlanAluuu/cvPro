import numpy as np
from scipy import signal
from PIL import Image
import matplotlib.pyplot as plt
import cv2


# 这个是添加噪声的函数，实际并没有用到
def noisy(imgpath):  
    # 加载图片
    # 使用Image.open()函数打开一张图片，将其转换为灰度图像(L 代表灰度模式Luminance)并将其转换为NumPy数组。
    img = np.array(Image.open(imgpath).convert('L'))

    # 添加高斯噪声
    mean = 0
    var = 200
    sigma = var ** 0.5
    # 生成指定均值和方差的高斯分布随机数，并将其加到图像数组中得到噪声图像
    gaussian_noise = np.random.normal(mean, sigma, img.shape)
    noisy_gaussian_image = img + gaussian_noise

    # 添加椒盐噪声
    noisy_salt_image = img
    # 设置椒盐噪声数量
    noise_num = 10000
    # 获取图像大小
    height, width = img.shape
    print(height,width)
    # 随机生成椒盐噪声位置
    x = np.random.randint(0,width,noise_num)
    y = np.random.randint(0,height,noise_num)
    print(x)
    # 将椒盐噪声添加到图像中
    for i in range(noise_num):
        if(y[i]<width & y[i]>0 & x[i]>0 & x[i]<height):
            noisy_salt_image[y[i],x[i]] = 0 # 黑色

    # 显示原始图片和带有噪声的图片
    fig, axes = plt.subplots(1, 3, figsize=(8, 4))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    axes[1].imshow(noisy_gaussian_image, cmap='gray')
    axes[1].set_title('Gaussian Noisy')
    axes[1].axis('off')
    axes[2].imshow(noisy_salt_image, cmap='gray')
    axes[2].set_title('Salt Noisy')
    axes[2].axis('off')
    plt.show()


# 生成二维高斯卷积核
# size表示卷积核的大小（假定卷积核是正方形），sigma表示高斯分布的标准差。
def gaussian_kernel(size, sigma):    
    # 计算出卷积核中心点的坐标
    center = size // 2    
    # 计算卷积核各元素值
    kernel = np.zeros([size, size])
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))

    # 归一化卷积核
    kernel /= np.sum(kernel)    
    return kernel

# 将生成的高斯卷积核应用于输入的图像上
def gaussian_filter(img, kernel_size=3, sigma=5):
    # 生成二维高斯卷积核
    kernel = gaussian_kernel(kernel_size, sigma)
    # 将图像与卷积核进行卷积操作
    filtered_image = signal.convolve2d(img, kernel, mode='same', boundary='symm')  
    return filtered_image


def sobel(img):
    # 定义Sobel算子模板
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    # 计算梯度
    grad_x = cv2.filter2D(img, -1, sobel_x)
    grad_y = cv2.filter2D(img, -1, sobel_y)

    # 缩放梯度值
    grad_x = cv2.convertScaleAbs(grad_x)
    grad_y = cv2.convertScaleAbs(grad_y)

    # 合并梯度图像
    grad = cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0)

    # theta
    theta = np.arctan2(grad_y, grad_x) 
    return grad_x, grad_y, grad ,theta



# 非极大值抑制算法：
# 1.对于每个像素点(x, y)，计算其梯度幅值G和梯度方向theta；
# 2.将theta转换为0到180度之间的整数角度；
# 3.判断当前像素点(x, y)处的梯度方向theta所对应的邻域像素的梯度值大小，如果当前像素点的梯度值不是该邻域像素中最大的，则将当前像素点的梯度值置为0，否则保留当前像素点的梯度值；
# 4.重复步骤3，直到遍历完所有的像素点。
def non_max_suppression(sobel_gx, sobel_gy, sobel_g, theta):
    rows, cols = sobel_g.shape
    sgn_theta = np.zeros((rows, cols))
    sgn_theta[(theta >= 0) & (theta <= 22.5)] = 0
    sgn_theta[(theta > 157.5) & (theta <= 180)] = 0
    sgn_theta[(theta > 22.5) & (theta <= 67.5)] = 45
    sgn_theta[(theta > 67.5) & (theta <= 112.5)] = 90
    sgn_theta[(theta > 112.5) & (theta <= 157.5)] = 135
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if sgn_theta[i, j] == 0:
                if sobel_g[i, j] < sobel_g[i, j-1] or sobel_g[i, j] < sobel_g[i, j+1]:
                    sobel_g[i, j] = 0
            elif sgn_theta[i, j] == 45:
                if sobel_g[i, j] < sobel_g[i-1, j+1] or sobel_g[i, j] < sobel_g[i+1, j-1]:
                    sobel_g[i, j] = 0
            elif sgn_theta[i, j] == 90:
                if sobel_g[i, j] < sobel_g[i-1, j] or sobel_g[i, j] < sobel_g[i+1, j]:
                    sobel_g[i, j] = 0
            elif sgn_theta[i, j] == 135:
                if sobel_g[i, j] < sobel_g[i-1, j-1] or sobel_g[i, j] < sobel_g[i+1, j+1]:
                    sobel_g[i, j] = 0
    return sobel_g

#连接与滞后阈值化
# 设置高、低两个阈值（一般高阈值是低阈值的2~3倍），遍历整个灰度矩阵，若某点的梯度高于高阈值，则在结果中置1，
# 若该点的梯度值低于低阈值，则在结果中置0，若该点的梯度值介于高低阈值之间，则需要进行如下判断：
# 检查该点的8邻域点，看是否存在梯度值高于高阈值的点，若存在，则说明该中心点和确定的边缘点相连接，
# 故在结果中置1，否则置0。
def edge_connection(grad, high, low):
    h, w = grad.shape
    # 初始化最终的边缘结果
    edge = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            if grad[i,j] >= high:
                edge[i,j] = 1
            elif grad[i,j] <= low:
                edge[i,j] = 0
            else:
                # 提取当前像素周围3x3的邻域矩阵current
                # 如果当前像素位于图像的边缘，则相应的邻域矩阵将自动被削减为合适的尺寸
                # max(0,i-1)确保了我们在i > 0时从i-1开始，而min(i+2,h)确保了我们不会将邻域超出图像的下边缘。
                # 同样，max(0,j-1)确保了我们在j > 0时从j-1开始，而min(j+2,w)则确保了我们不会将邻域超出图像的右边缘。
                current = grad[max(0,i-1):min(i+1,h), max(0,j-1):min(j+1,w)]
                maxvalue = np.max(current)
                if maxvalue >= high:
                    edge[i,j] = 1
                else:
                    edge[i,j] = 0
    return edge



# 
# 第一步，高斯滤波
#
img = np.array(Image.open("1.Canny边缘检测\img\hutao.jpg").convert('L'))
histr = cv2.calcHist([img],[0],None,[256],[0,256])#灰度图
#cv2.calcHist()函数计算的是图像的直方图数据，而不是图像本身
#因此，在使用plt.imshow()函数显示直方图时，需要注意一些问题
plt.figure(figsize=(10,6),dpi=100)
plt.plot(histr)
plt.grid()
plt.show()
gaussian_filtered_image=gaussian_filter(img, kernel_size=3, sigma=5)
plt.imshow(gaussian_filtered_image, cmap='gray')
plt.title("gaussian_filtered_image")
plt.show()

# 第二步，利用Sobel算子计算梯度信息
Sobel_Gx, Sobel_Gy, Sobel_G, theta=sobel(gaussian_filtered_image)
# 创建一个1x3的网格，用于显示3张图像
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8,6))
# cmap=plt.cm.gray将图像以灰度图像的形式显示，
# 即使用黑白灰色调表示图像中的像素强度值。如果不使用该参数，则默认使用彩色图像显示。
axes[0].imshow(Sobel_Gx,cmap=plt.cm.gray)
axes[0].set_title("Sobel_Gx")
axes[1].imshow(Sobel_Gy,cmap=plt.cm.gray)
axes[1].set_title("Sobel_Gy")
axes[2].imshow(Sobel_G,cmap=plt.cm.gray)
axes[2].set_title("Sobel_G")
plt.show()

# 第三步，非极大值抑制(NMS)
NMS_img= non_max_suppression(Sobel_Gx,Sobel_Gy,Sobel_G,theta)
plt.imshow(NMS_img, cmap='gray')
plt.title("NMS_img")
plt.show()

# 第四步，连接与滞后阈值化
# 设置阈值
high=80
low=40
edg_img = edge_connection(NMS_img, high, low)

# 内置函数实现Canny算子检测
canny_img = cv2.Canny(img, high, low)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,6))
axes[0].imshow(edg_img,cmap=plt.cm.gray)
axes[0].set_title("edg_img")
axes[1].imshow(canny_img,cmap=plt.cm.gray)
axes[1].set_title("canny_img")
plt.show()