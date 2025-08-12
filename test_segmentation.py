from predict_segmentation import Predict_Segmentation

predict = Predict_Segmentation(model_path='/Users/peteksener/Desktop/y/iris.pth')
image_path = "/Users/peteksener/Desktop/train/img/0214_2_1_2_33_004.png"  
image, tensor = predict.preprocess_image(image_path)
mask1 = predict.predict_mask(predict.model, tensor)

predict.visualize_segmentation(image, image, mask1)

