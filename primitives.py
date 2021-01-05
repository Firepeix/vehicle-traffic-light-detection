import tensorflow
from PIL import Image

class ImagePrimitive:
    def __init__(self, path):
        if path != 'tmp':
            self.__path = path
            self.__image = tensorflow.keras.preprocessing.image.img_to_array(Image.open(self.__path))
            self.__image = tensorflow.cast(self.__image, dtype=tensorflow.float32)
            self.__input_image = None
            self.__ratio = None

    @staticmethod
    def from_request(image):
        image_primitive = ImagePrimitive('tmp')
        image_primitive.__path = 'tmp'
        image_primitive.__image = tensorflow.keras.preprocessing.image.img_to_array(image)
        image_primitive.__image = tensorflow.cast(image_primitive.__image, dtype=tensorflow.float32)
        image_primitive.__input_image = None
        image_primitive.__ratio = None
        return image_primitive

    def get_raw(self):
        return self.__image

    def get_input_image(self):
        return self.__input_image

    def get_ratio(self):
        return self.__ratio

    def prepare_image(self):
        self.__input_image, self.__ratio = self.general_prepare_image(self.__image)

    @staticmethod
    def general_prepare_image(image):
        image, _, ratio = ImagePrimitive.general_resize_and_pad_image(image, jitter=None)
        image = tensorflow.keras.applications.resnet.preprocess_input(image)
        return tensorflow.expand_dims(image, axis=0), ratio

    @staticmethod
    def general_resize_and_pad_image(
            image, min_side=800.0, max_side=1333.0, jitter=[640, 1024], stride=128.0
    ):
        """Resizes and pads image while preserving aspect ratio.

        1. Resizes images so that the shorter side is equal to `min_side`
        2. If the longer side is greater than `max_side`, then resize the image
          with longer side equal to `max_side`
        3. Pad with zeros on right and bottom to make the image shape divisible by
        `stride`

        Arguments:
          image: A 3-D tensor of shape `(height, width, channels)` representing an
            image.
          min_side: The shorter side of the image is resized to this value, if
            `jitter` is set to None.
          max_side: If the longer side of the image exceeds this value after
            resizing, the image is resized such that the longer side now equals to
            this value.
          jitter: A list of floats containing minimum and maximum size for scale
            jittering. If available, the shorter side of the image will be
            resized to a random value in this range.
          stride: The stride of the smallest feature map in the feature pyramid.
            Can be calculated using `image_size / feature_map_size`.

        Returns:
          image: Resized and padded image.
          image_shape: Shape of the image before padding.
          ratio: The scaling factor used to resize the image
        """
        image_shape = tensorflow.cast(tensorflow.shape(image)[:2], dtype=tensorflow.float32)
        if jitter is not None:
            min_side = tensorflow.random.uniform((), jitter[0], jitter[1], dtype=tensorflow.float32)
        ratio = min_side / tensorflow.reduce_min(image_shape)
        if ratio * tensorflow.reduce_max(image_shape) > max_side:
            ratio = max_side / tensorflow.reduce_max(image_shape)
        image_shape = ratio * image_shape
        image = tensorflow.image.resize(image, tensorflow.cast(image_shape, dtype=tensorflow.int32))
        padded_image_shape = tensorflow.cast(
            tensorflow.math.ceil(image_shape / stride) * stride, dtype=tensorflow.int32
        )
        image = tensorflow.image.pad_to_bounding_box(
            image, 0, 0, padded_image_shape[0], padded_image_shape[1]
        )
        return image, image_shape, ratio
