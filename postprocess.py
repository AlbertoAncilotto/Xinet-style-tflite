import cv2
import os
import numpy as np

def increase_contrast(image, alpha=1.25, beta=-10):
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

def increase_saturation(image, saturation_scale=1.2):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image = hsv_image.astype('float32')
    hsv_image[:, :, 1] *= saturation_scale
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1], 0, 255)
    hsv_image = hsv_image.astype('uint8')
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

def apply_bilateral_filter(image, diameter=15, sigma_color=20, sigma_space=10):
    filtered_image = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
    return filtered_image

def increase_vibrance(image, vibrance_scale=1):
    # Convert to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image = hsv_image.astype('float32')
    
    # Increase vibrance
    for i in range(hsv_image.shape[0]):
        for j in range(hsv_image.shape[1]):
            h, s, v = hsv_image[i, j]
            increase = vibrance_scale * (1 - s / 255.0)
            s += s * increase
            s = min(255, max(0, s))
            hsv_image[i, j][1] = s
    
    hsv_image = hsv_image.astype('uint8')
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

def remove_color_noise(image):
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv_image)
    
    u = cv2.fastNlMeansDenoising(u, None, 5, 7, 21)
    v = cv2.fastNlMeansDenoising(v, None, 5, 7, 21)
    
    denoised_yuv = cv2.merge([y, u, v])
    denoised_image = cv2.cvtColor(denoised_yuv, cv2.COLOR_YUV2BGR)
    return denoised_image

def process_frame(image, show=False):
    filtered_image = apply_bilateral_filter(image)
    denoised_image = remove_color_noise(filtered_image)
    sat_img = increase_saturation(denoised_image)
    final_image = increase_contrast(sat_img)
    
    if show:
        # Show the processed image
        cv2.imshow('Original Image', image)
        cv2.imshow('Denoise', denoised_image)
        cv2.imshow('Bilateral filt', filtered_image)
        cv2.imshow('Sat', sat_img)

    return final_image

def process_images(folder_path):
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)

        final_image = process_frame(image)
        cv2.imshow('Processed Image', final_image)
        
        key = cv2.waitKey(0)
        if key == 27:  # ESC to exit
            break
    
    cv2.destroyAllWindows()

if __name__=='__main__':
    folder_path = 'demo_images'
    process_images(folder_path)
