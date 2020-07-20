from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
from matplotlib.image import imread
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt

prediction_key = "YOUR PREDICTION KEY HERE" 
ENDPOINT = "YOUR ENDPOINT HERE"
project_id = "YOUR PROJECT ID HERE"
publish_iteration_name = "YOUR ITERATION NAME HERE"

prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)

with open("URL TO THE IMAGE", mode="rb") as test_data:
    results = predictor.detect_image(project_id, publish_iteration_name, test_data)

img = imread("URL TO THE IMAGE")
alto_imagen, ancho_imagen, _ = img.shape 
_ , ax = plt.subplots()
ax.axis("off")
ax.imshow(img)
for prediction in results.predictions:
    print("\t" + prediction.tag_name + """": {0:.2f}% bbox.left = {1:.2f}, bbox.top = {2:.2f},
    bbox.width = {3:.2f}, bbox.height = {4:.2f}""".format(prediction.probability * 100, prediction.bounding_box.left,
    prediction.bounding_box.top, prediction.bounding_box.width, prediction.bounding_box.height))
    if prediction.probability * 100 > 60.0:
        rect = Rectangle((prediction.bounding_box.left * ancho_imagen,
                          prediction.bounding_box.top * alto_imagen),
                          prediction.bounding_box.width * ancho_imagen,
                          prediction.bounding_box.height * alto_imagen, 
                          edgecolor = 'r', facecolor = "none")
        ax.add_patch(rect)
plt.title("Resultados")
plt.show()

