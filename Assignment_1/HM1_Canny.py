import numpy as np
from HM1_Convolve import Gaussian_filter, Sobel_filter_x, Sobel_filter_y
from utils import read_img, write_img

def compute_gradient_magnitude_direction(x_grad, y_grad):
    """
        The function you need to implement for Q2 a).
        Inputs:
            x_grad: array(float) 
            y_grad: array(float)
        Outputs:
            magnitude_grad: array(float)
            direction_grad: array(float) you may keep the angle of the gradient at each pixel
    """
    magnitude_grad = np.sqrt(x_grad**2 + y_grad**2)
    direction_grad = np.arctan2(y_grad, x_grad)
    return magnitude_grad, direction_grad 



def non_maximal_suppressor(grad_mag, grad_dir):
    """
        The function you need to implement for Q2 b).
        Inputs:
            grad_mag: array(float) 
            grad_dir: array(float)
        Outputs:
            output: array(float)
    """   
    #采用简化做法，只考虑沿梯度正反向的两个像素
    #将角度从弧度转换为角度，并将角度转换到[0,180)的范围
    grad_dir = (grad_dir/np.pi*180)%180
    
    #分为四个角度区间，在每个区间方向上进行正反向的比较
    east_sector = (grad_dir >= 157.5) | (grad_dir < 22.5)
    northeast_sector = (grad_dir >= 22.5) & (grad_dir < 67.5)
    north_sector = (grad_dir >= 67.5) & (grad_dir < 112.5)
    northwest_sector = (grad_dir >= 112.5) & (grad_dir < 157.5)

    east_shift = np.roll(grad_mag, shift=-1, axis=1)
    west_shift = np.roll(grad_mag, shift=1, axis=1)
    north_shift = np.roll(grad_mag, shift=-1, axis=0)
    south_shift = np.roll(grad_mag, shift=1, axis=0)
    northeast_shift = np.roll(north_shift, shift=-1, axis=1)
    southwest_shift = np.roll(south_shift, shift=1, axis=1)
    northwest_shift = np.roll(north_shift, shift=1, axis=1)
    southeast_shift = np.roll(south_shift, shift=-1, axis=1)
    
    NMS_output = np.zeros_like(grad_mag)
    NMS_output[east_sector & (grad_mag >= east_shift) & (grad_mag >= west_shift)] = grad_mag[east_sector & (grad_mag >= east_shift) & (grad_mag >= west_shift)]
    NMS_output[northeast_sector & (grad_mag >= northeast_shift) & (grad_mag >= southwest_shift)] = grad_mag[northeast_sector & (grad_mag >= northeast_shift) & (grad_mag >= southwest_shift)]
    NMS_output[north_sector & (grad_mag >= north_shift) & (grad_mag >= south_shift)] = grad_mag[north_sector & (grad_mag >= north_shift) & (grad_mag >= south_shift)]
    NMS_output[northwest_sector & (grad_mag >= northwest_shift) & (grad_mag >= southeast_shift)] = grad_mag[northwest_sector & (grad_mag >= northwest_shift) & (grad_mag >= southeast_shift)]

    return NMS_output 
            


def hysteresis_thresholding(img) :
    """
        The function you need to implement for Q2 c).
        Inputs:
            img: array(float) 
        Outputs:
            output: array(float)
    """


    #you can adjust the parameters to fit your own implementation 
    low_ratio = 0.07
    high_ratio = 0.17
    
    H, W = img.shape
    
    maxVal =  high_ratio * np.max(img) #upper threshold
    minVal = low_ratio * maxVal #lower threshold
    
    strong_edge = (img > maxVal)
    weak_edge = (img > minVal) & (img <= maxVal)
    output = np.zeros_like(img)
    #初始化output将strong edge置为1
    output[strong_edge] = 1 
    
    #对weak edge进行连接
    linked = True
    while linked:
        linked = False
        for i in range(1, H-1):
            for j in range(1, W-1):
                if weak_edge[i, j]:
                    if np.any(strong_edge[i-1:i+2, j-1:j+2]):
                        output[i, j] = 1
                        strong_edge[i, j] = 1
                        weak_edge[i, j] = 0
    
    return output 



if __name__=="__main__":

    #Load the input images
    input_img = read_img("Lenna.png")/255

    #Apply gaussian blurring
    blur_img = Gaussian_filter(input_img)

    x_grad = Sobel_filter_x(blur_img)
    y_grad = Sobel_filter_y(blur_img)

    #Compute the magnitude & the direction of gradient
    magnitude_grad, direction_grad = compute_gradient_magnitude_direction(x_grad, y_grad)

    #NMS
    NMS_output = non_maximal_suppressor(magnitude_grad, direction_grad)

    #Edge linking with hysteresis
    output_img = hysteresis_thresholding(NMS_output)
    
    write_img("result/HM1_Canny_result.png", output_img*255)
