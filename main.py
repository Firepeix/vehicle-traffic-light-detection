import matplotlib.pyplot as plt
from core import *
from primitives import *

def detect_in_image ():
    # detector = Detector('retinanet/')
    detector = Detector('data', 80)
    detector.create_model(get_backbone())
    detector.load_dataset('data', 'coco/2017')
    detector.load_weights()
    detector.create_inference_model()
    image = ImagePrimitive('1.jpeg')
    image.prepare_image()
    detections = detector.predict(image.get_input_image())
    num_detections = detections.valid_detections[0]
    class_names = detector.get_class_names(detections)
    visualize_detections(
        image.get_raw(),
        detections.nmsed_boxes[0][:num_detections] / image.get_ratio(),
        class_names,
        detections.nmsed_scores[0][:num_detections],
    )

def train_neural_network():
    trainer = Trainer('data', 'test', 80)
    trainer.create_model(get_backbone())
    trainer.compile()
    trainer.load_dataset('coco/2017')
    trainer.prepare_train_dataset()
    trainer.prepare_validation_dataset()
    trainer.fit()

if __name__ == '__main__':
    detect_in_image()