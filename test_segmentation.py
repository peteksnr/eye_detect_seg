from predict_segmentation import Predict_Segmentation

predict = Predict_Segmentation(model_path="path/to/segmentation.model")
image_path = "path/to/image"
image, tensor = predict.preprocess_image(image_path)
mask1 = predict.predict_mask(predict.model, tensor)

predict.visualize_segmentation(image, image, mask1)

