import cv2
import albumentations as A
import random
import os

hsv_h = random.randint(0, 20)
hsv_s = random.randint(-20, 20)
hsv_v = random.randint(0, 20) 
brightness_limit= random.uniform(0.1,0.2)
contrast_limit= random.random()
degrees = random.randint(-5, 5) 
translate = random.uniform(-0.1, 0.1) 
scale = random.uniform(-0.1, 0.1)
shear = (random.randint(-5, 5), random.randint(-5, 5)  )
perspective = 0.0 
fliplr = 0.5  
 

image_dir = '/Users/peteksener/Desktop/v/i'
mask_dir = '/Users/peteksener/Desktop/v/m'
output_image_dir = '/Users/peteksener/Desktop/untitled folder 2/i'
output_mask_dir = '/Users/peteksener/Desktop/untitled folder 2/m'


transform = A.Compose([
    A.HorizontalFlip(p=fliplr),  
    A.RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=contrast_limit, p=0.4),
    A.HueSaturationValue(hue_shift_limit=hsv_h, sat_shift_limit=hsv_s, val_shift_limit=hsv_v, p=0.4),
    A.Rotate(limit=degrees, p=0.5),  
    # A.ShiftScaleRotate(shift_limit=translate, scale_limit=scale, rotate_limit=degrees, p=0.5),
    A.Affine(scale=1.2, shear=shear, p=0.3),  
])


    
count = 10
def process_dataset(image_dir, mask_dir, output_image_dir, output_mask_dir):
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])   
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        mask_path = os.path.join(mask_dir, image_file.replace('.png', '.png'))
        try:
            for i in range(count):
                output_image_path = os.path.join(output_image_dir, f'{str(image_file).replace(".png", f"_{i}.png")}')
                output_mask_path = os.path.join(output_mask_dir, f'{str(image_file).replace(".png", f"_{i}.png")}')
                image = cv2.imread(image_path)
                mask = cv2.imread(mask_path)
                transformed = transform(image=image, mask=mask)
                augmented_image = transformed['image'] 
                augmented_mask = transformed['mask']
                cv2.imwrite(output_image_path, augmented_image)
                cv2.imwrite(output_mask_path, augmented_mask)
                print('succes')
        except:
            print('error')

process_dataset(image_dir, mask_dir, output_image_dir, output_mask_dir)