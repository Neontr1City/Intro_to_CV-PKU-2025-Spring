import numpy as np
from utils import read_img, write_img

def padding(img, padding_size, type):
    """
        The function you need to implement for Q1 a).
        Inputs:
            img: array(float)
            padding_size: int
            type: str, zeroPadding/replicatePadding
        Outputs:
            padding_img: array(float)
    """

    if type=="zeroPadding":
        hei, wid = img.shape
        padding_img = np.zeros((hei+2*padding_size, wid+2*padding_size))
        
        padding_img[padding_size:hei+padding_size, padding_size:wid+padding_size] = img
        
        return padding_img
    elif type=="replicatePadding":
        h, w = img.shape
        p = padding_size
        padding_img = np.zeros((h + 2*p, w + 2*p))
        
        padding_img[p:h+p, p:w+p] = img
        padding_img[0:p, p:w+p] = img[0:1, :]
        padding_img[h+p:, p:w+p] = img[-1:, :]
        padding_img[:, 0:p] = padding_img[:, p:p+1]
        padding_img[:, w+p:] = padding_img[:, w+p-1:w+p]

        return padding_img


def convol_with_Toeplitz_matrix(img, kernel):
    """
        The function you need to implement for Q1 b).
        Inputs:
            img: array(float) 6*6
            kernel: array(float) 3*3
        Outputs:
            output: array(float)
    """
    #zero padding
    padding_img = padding(img, 1, "zeroPadding") #8*8

    #build the Toeplitz matrix and compute convolution
    idx = np.arange(0, 64).reshape(8, 8) #8*8的下标矩阵从0到63
    #从idx中取出9个6*6的滑动窗口，分别对应kernel的9个元素的列索引
    kernel0_idx = idx[0:6, 0:6].flatten() 
    kernel1_idx = idx[0:6, 1:7].flatten()
    kernel2_idx = idx[0:6, 2:8].flatten()
    kernel3_idx = idx[1:7, 0:6].flatten()
    kernel4_idx = idx[1:7, 1:7].flatten()
    kernel5_idx = idx[1:7, 2:8].flatten()
    kernel6_idx = idx[2:8, 0:6].flatten()
    kernel7_idx = idx[2:8, 1:7].flatten()
    kernel8_idx = idx[2:8, 2:8].flatten()
    
    col_idx = np.array([kernel0_idx, kernel1_idx, kernel2_idx, kernel3_idx, kernel4_idx, kernel5_idx, kernel6_idx, kernel7_idx, kernel8_idx])
    row_idx = np.tile(np.arange(36), (9, 1)) #行索引矩阵，9*(0-35)
    
    kernel_flat = kernel.flatten()
    #the Toeplitz matrix should be 36*64
    T = np.zeros((36, 64))
    T[row_idx, col_idx] = kernel_flat[:, None]

    
    #output should be 6*6
    output = T @ padding_img.flatten()
    output = output.reshape(6, 6)
    return output


def convolve(img, kernel):
    """
        The function you need to implement for Q1 c).
        Inputs:
            img: array(float)
            kernel: array(float)
        Outputs:
            output: array(float)
    """
    
    #build the sliding-window convolution here
    H, W = img.shape
    k = kernel.shape[0]
    out_h = H - k + 1
    out_w = W - k + 1
    
    idx = np.arange(0, H*W).reshape(H, W)
    #用meshgrid构造滑动窗口左上角的索引
    row_idx, col_idx = np.meshgrid(np.arange(out_h), np.arange(out_w), indexing='ij')
    top_left_idx = (row_idx * W + col_idx).ravel()
    
    #用meshgrid构造滑动窗口内部的偏移索引
    row_offset, col_offset = np.meshgrid(np.arange(k), np.arange(k), indexing='ij')
    offset_idx = (row_offset * W + col_offset).ravel()
    
    #用索引构造滑动窗口的索引矩阵
    window_idx = top_left_idx[:, None] + offset_idx[None, :] #利用broadcasting构建(out_h*out_w)*(k*k)的索引矩阵
    #根据索引矩阵取出滑动窗口
    windows = img.ravel()[window_idx]
    
    #计算卷积
    conv = windows @ kernel.ravel()
    output = conv.reshape(out_h, out_w)

    return output


def Gaussian_filter(img):
    padding_img = padding(img, 1, "replicatePadding")
    gaussian_kernel = np.array([[1/16,1/8,1/16],[1/8,1/4,1/8],[1/16,1/8,1/16]])
    output = convolve(padding_img, gaussian_kernel)
    return output

def Sobel_filter_x(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    output = convolve(padding_img, sobel_kernel_x)
    return output

def Sobel_filter_y(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    output = convolve(padding_img, sobel_kernel_y)
    return output



if __name__=="__main__":

    np.random.seed(111)
    input_array=np.random.rand(6,6)
    input_kernel=np.random.rand(3,3)


    # task1: padding
    zero_pad =  padding(input_array,1,"zeroPadding")
    np.savetxt("result/HM1_Convolve_zero_pad.txt",zero_pad)

    replicate_pad = padding(input_array,1,"replicatePadding")
    np.savetxt("result/HM1_Convolve_replicate_pad.txt",replicate_pad)


    #task 2: convolution with Toeplitz matrix
    result_1 = convol_with_Toeplitz_matrix(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_1.txt", result_1)

    #task 3: convolution with sliding-window
    result_2 = convolve(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_2.txt", result_2)

    #task 4/5: Gaussian filter and Sobel filter
    input_img = read_img("lenna.png")/255

    img_gadient_x = Sobel_filter_x(input_img)
    img_gadient_y = Sobel_filter_y(input_img)
    img_blur = Gaussian_filter(input_img)

    write_img("result/HM1_Convolve_img_gadient_x.png", img_gadient_x*255)
    write_img("result/HM1_Convolve_img_gadient_y.png", img_gadient_y*255)
    write_img("result/HM1_Convolve_img_blur.png", img_blur*255)




    