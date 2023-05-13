import numpy as np
import gradio as gr

def sobel(input_img):

    lumin = input_img[:, :, 0] * 0.21 + input_img[:, :, 1] * 0.72 + input_img[:, :, 2] * 0.07
    
    gx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]])
    
    gy = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]])
    
    gx_img = np.zeros_like(lumin)
    gy_img = np.zeros_like(lumin)
    
    axes = input_img.shape
    cols = axes[0]
    rows = axes[1]
    channels = axes[2]
    
    
    for i in range(rows)[1: -2]:
        for j in range(cols)[1: -2]:
                gx_img[i, j] = np.sum(gx * lumin[i: i + 3, j: j + 3])
                gy_img[i, j] = np.sum(gy * lumin[i: i + 3, j: j + 3])
           
    sobel_img = np.sqrt(gx_img * gx_img + gy_img * gy_img)
    sobel_img = sobel_img / sobel_img.max() * 255
    return sobel_img.astype(np.uint8)

demo = gr.Interface(
    sobel,
    gr.Image(shape=(256, 256)),
    "image",
)

demo.launch(server_port=8080)