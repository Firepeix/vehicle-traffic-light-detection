import flask
from core import *
from primitives import *
import io

app = flask.Flask(__name__)
detector = None

def load_detector():
    global detector
    detector = Detector('data', 80)
    detector.create_model(get_backbone())
    detector.load_dataset('data', 'coco/2017')
    detector.load_weights()
    detector.create_inference_model()


def prepare_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = ImagePrimitive.from_request(image)
    image.prepare_image()
    return image

@app.route("/predict", methods=["POST"])

def predict():
    data = {"success": False}
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
          image = flask.request.files["image"].read()
          image = Image.open(io.BytesIO(image))
          image = prepare_image(image)

          detections = detector.predict(image.get_input_image())
          num_detections = int(detections.valid_detections[0])
          class_names = detector.get_class_names(detections)
          data["predictions"] = []
          data["totalObjectsFound"] = num_detections

          boxes = detections.nmsed_boxes[0][:num_detections] / image.get_ratio()


          for detection in range(0, num_detections):
              found = {
                  "label": class_names[detection],
                  'probability': detections.nmsed_scores[0][detection] * 100,
                  "box": boxes[detection].numpy().tolist()
              }
              data["predictions"].append(found)
          data["success"] = True
    return flask.make_response(flask.jsonify(data), 200)

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_detector()
    app.run(host='0.0.0.0')
