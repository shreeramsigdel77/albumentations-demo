import json
import streamlit as st
import cv2
import os

aug_value = {
    'OneOf':
     [
         {
            'Blur': [{'defaults': [3, 7], 'limits_list': [3, 100], 'param_name': 'blur_limit', 'type': 'num_interval'}],
            'CLAHE': [{'defaults': [1, 4], 'limits_list': [1, 100], 'param_name': 'clip_limit', 'type': 'num_interval'}, {'defaults_list': [8, 8], 'limits_list': [[1, 100], [1, 100]], 'param_name': 'tile_grid_size', 'subparam_names': ['height', 'width'], 'type': 'several_nums'}],
            'CenterCrop': [{'param_name': 'height', 'placeholder': {'defaults': 'image_half_height', 'limits_list': [1, 'image_height']}, 'type': 'num_interval'}, {'param_name': 'width', 'placeholder': {'defaults': 'image_half_width', 'limits_list': [1, 'image_width']}, 'type': 'num_interval'}]
            }
         ], 
         'Blur': [{'defaults': [3, 7], 'limits_list': [3, 100], 'param_name': 'blur_limit', 'type': 'num_interval'}], 
         'CLAHE': [{'defaults': [1, 4], 'limits_list': [1, 100], 'param_name': 'clip_limit', 'type': 'num_interval'}, {'defaults_list': [8, 8], 'limits_list': [[1, 100], [1, 100]], 'param_name': 'tile_grid_size', 'subparam_names': ['height', 'width'], 'type': 'several_nums'}], 
         'CenterCrop': [{'param_name': 'height', 'type': 'num_interval', 'defaults': 106, 'limits_list': [1, 212]}, {'param_name': 'width', 'type': 'num_interval', 'defaults': 160, 'limits_list': [1, 320]}], 
         'ChannelDropout': [{'defaults': [1, 1], 'limits_list': [1, 2], 'param_name': 'channel_drop_range', 'type': 'num_interval'}, {'defaults': 0, 'limits_list': [0, 255], 'param_name': 'fill_value', 'type': 'num_interval'}], 
         'ChannelShuffle': [], 
         'CoarseDropout': [{'defaults_list': [8, 8], 'limits_list': [1, 100], 'min_diff': 0, 'param_name': ['min_holes', 'max_holes'], 'type': 'min_max'}, {'defaults_list': [8, 8], 'limits_list': [1, 100], 'min_diff': 0, 'param_name': ['min_height', 'max_height'], 'type': 'min_max'}, {'defaults_list': [8, 8], 'limits_list': [1, 100], 'min_diff': 0, 'param_name': ['min_width', 'max_width'], 'type': 'min_max'}, {'param_name': 'fill_value', 'type': 'rgb'}], 
         'Crop': [{'min_diff': 1, 'param_name': ['x_min', 'x_max'], 'type': 'min_max', 'defaults_list': [0, 160], 'limits_list': [0, 320]}, {'min_diff': 1, 'param_name': ['y_min', 'y_max'], 'type': 'min_max', 'defaults_list': [0, 106], 'limits_list': [0, 212]}], 'Cutout': [{'defaults': 8, 'limits_list': [1, 100], 'param_name': 'num_holes', 'type': 'num_interval'}, {'defaults': 8, 'limits_list': [1, 100], 'param_name': 'max_h_size', 'type': 'num_interval'}, {'defaults': 8, 'limits_list': [1, 100], 'param_name': 'max_w_size', 'type': 'num_interval'}, {'param_name': 'fill_value', 'type': 'rgb'}], 'Downscale': [{'defaults_list': [0.25, 0.25], 'limits_list': [0.01, 0.99], 'param_name': ['scale_min', 'scale_max'], 'type': 'min_max'}, {'options_list': [0, 1, 2, 3, 4], 'param_name': 'interpolation', 'type': 'radio'}], 'ElasticTransform': [{'defaults': 1.0, 'limits_list': [0.0, 10.0], 'param_name': 'alpha', 'type': 'num_interval'}, {'defaults': 50.0, 'limits_list': [0.0, 200.0], 'param_name': 'sigma', 'type': 'num_interval'}, {'defaults': 50.0, 'limits_list': [0.0, 200.0], 'param_name': 'alpha_affine', 'type': 'num_interval'}, {'options_list': [0, 1, 2, 3, 4], 'param_name': 'interpolation', 'type': 'radio'}, {'options_list': [0, 1, 2, 3, 4], 'param_name': 'border_mode', 'type': 'radio'}, {'param_name': 'value', 'type': 'rgb'}], 'Equalize': [{'options_list': ['cv', 'pil'], 'param_name': 'mode', 'type': 'radio'}, {'defaults': 1, 'param_name': 'by_channels', 'type': 'checkbox'}], 'Flip': [], 'GaussNoise': [{'defaults': [10.0, 50.0], 'limits_list': [0.0, 500.0], 'param_name': 'var_limit', 'type': 'num_interval'}, {'defaults': 0.0, 'limits_list': [-100.0, 100.0], 'param_name': 'mean', 'type': 'num_interval'}], 'GridDistortion': [{'defaults': 5, 'limits_list': [1, 15], 'param_name': 'num_steps', 'type': 'num_interval'}, {'defaults': [-0.3, 0.3], 'limits_list': [-2.0, 2.0], 'param_name': 'distort_limit', 'type': 'num_interval'}, {'options_list': [0, 1, 2, 3, 4], 'param_name': 'interpolation', 'type': 'radio'}, {'options_list': [0, 1, 2, 3, 4], 'param_name': 'border_mode', 'type': 'radio'}, {'param_name': 'value', 'type': 'rgb'}], 'HorizontalFlip': [], 'HueSaturationValue': [{'defaults': [-20, 20], 'limits_list': [-100, 100], 'param_name': 'hue_shift_limit', 'type': 'num_interval'}, {'defaults': [-30, 30], 'limits_list': [-100, 100], 'param_name': 'sat_shift_limit', 'type': 'num_interval'}, {'defaults': [-20, 20], 'limits_list': [-100, 100], 'param_name': 'val_shift_limit', 'type': 'num_interval'}], 'ISONoise': [{'defaults': [0.01, 0.05], 'limits_list': [0.0, 1.0], 'param_name': 'color_shift', 'type': 'num_interval'}, {'defaults': [0.1, 0.5], 'limits_list': [0.0, 2.0], 'param_name': 'intensity', 'type': 'num_interval'}], 'ImageCompression': [{'options_list': [0, 1], 'param_name': 'compression_type', 'type': 'radio'}, {'defaults_list': [80, 100], 'limits_list': [0, 100], 'param_name': ['quality_lower', 'quality_upper'], 'type': 'min_max'}], 'InvertImg': [], 'JpegCompression': [{'defaults_list': [80, 100], 'limits_list': [0, 100], 'param_name': ['quality_lower', 'quality_upper'], 'type': 'min_max'}], 'LongestMaxSize': [{'defaults': 512, 'limits_list': [1, 1024], 'param_name': 'max_size', 'type': 'num_interval'}, {'options_list': [0, 1, 2, 3, 4], 'param_name': 'interpolation', 'type': 'radio'}], 'MotionBlur': [{'defaults': [3, 7], 'limits_list': [3, 100], 'param_name': 'blur_limit', 'type': 'num_interval'}], 'MultiplicativeNoise': [{'defaults': [0.9, 1.1], 'limits_list': [0.1, 5.0], 'param_name': 'multiplier', 'type': 'num_interval'}, {'defaults': 1, 'param_name': 'per_channel', 'type': 'checkbox'}, {'defaults': 1, 'param_name': 'elementwise', 'type': 'checkbox'}], 'OpticalDistortion': [{'defaults': [-0.3, 0.3], 'limits_list': [-2.0, 2.0], 'param_name': 'distort_limit', 'type': 'num_interval'}, {'defaults': [-0.05, 0.05], 'limits_list': [-1.0, 1.0], 'param_name': 'shift_limit', 'type': 'num_interval'}, {'options_list': [0, 1, 2, 3, 4], 'param_name': 'interpolation', 'type': 'radio'}, {'options_list': [0, 1, 2, 3, 4], 'param_name': 'border_mode', 'type': 'radio'}, {'param_name': 'value', 'type': 'rgb'}], 'Posterize': [{'defaults_list': [4, 4, 4], 'limits_list': [[0, 8], [0, 8], [0, 8]], 'param_name': 'num_bits', 'subparam_names': ['r', 'g', 'b'], 'type': 'several_nums'}], 'RGBShift': [{'defaults': [-20, 20], 'limits_list': [-255, 255], 'param_name': 'r_shift_limit', 'type': 'num_interval'}, {'defaults': [-20, 20], 'limits_list': [-255, 255], 'param_name': 'g_shift_limit', 'type': 'num_interval'}, {'defaults': [-20, 20], 'limits_list': [-255, 255], 'param_name': 'b_shift_limit', 'type': 'num_interval'}], 'RandomBrightness': [{'defaults': [-0.2, 0.2], 'limits_list': [-1.0, 1.0], 'param_name': 'limit', 'type': 'num_interval'}], 'RandomBrightnessContrast': [{'defaults': [-0.2, 0.2], 'limits_list': [-1.0, 1.0], 'param_name': 'brightness_limit', 'type': 'num_interval'}, {'defaults': [-0.2, 0.2], 'limits_list': [-1.0, 1.0], 'param_name': 'contrast_limit', 'type': 'num_interval'}, {'defaults': 1, 'param_name': 'brightness_by_max', 'type': 'checkbox'}], 'RandomContrast': [{'defaults': [-0.2, 0.2], 'limits_list': [-1.0, 1.0], 'param_name': 'limit', 'type': 'num_interval'}], 'RandomFog': [{'defaults_list': [0.1, 0.2], 'limits_list': [0.0, 1.0], 'param_name': ['fog_coef_lower', 'fog_coef_upper'], 'type': 'min_max'}, {'defaults': 0.08, 'limits_list': [0.0, 1.0], 'param_name': 'alpha_coef', 'type': 'num_interval'}], 'RandomGamma': [{'defaults': [80, 120], 'limits_list': [0, 200], 'param_name': 'gamma_limit', 'type': 'num_interval'}], 'RandomGridShuffle': [{'defaults_list': [3, 3], 'limits_list': [[1, 10], [1, 10]], 'param_name': 'grid', 'subparam_names': ['vertical', 'horizontal'], 'type': 'several_nums'}], 'RandomRain': [{'defaults_list': [-10, 10], 'limits_list': [-20, 20], 'param_name': ['slant_lower', 'slant_upper'], 'type': 'min_max'}, {'defaults': 20, 'limits_list': [0, 100], 'param_name': 'drop_length', 'type': 'num_interval'}, {'defaults': 1, 'limits_list': [1, 5], 'param_name': 'drop_width', 'type': 'num_interval'}, {'param_name': 'drop_color', 'type': 'rgb'}, {'defaults': 7, 'limits_list': [1, 15], 'param_name': 'blur_value', 'type': 'num_interval'}, {'defaults': 0.7, 'limits_list': [0.0, 1.0], 'param_name': 'brightness_coefficient', 'type': 'num_interval'}, {'options_list': ['None', 'drizzle', 'heavy', 'torrential'], 'param_name': 'rain_type', 'type': 'radio'}], 'RandomResizedCrop': [{'param_name': 'height', 'type': 'num_interval', 'defaults': 212, 'limits_list': [1, 212]}, {'param_name': 'width', 'type': 'num_interval', 'defaults': 320, 'limits_list': [1, 320]}, {'defaults': [0.08, 1.0], 'limits_list': [0.01, 1.0], 'param_name': 'scale', 'type': 'num_interval'}, {'defaults': [0.75, 1.3333333333333333], 'limits_list': [0.1, 10.0], 'param_name': 'ratio', 'type': 'num_interval'}, {'options_list': [0, 1, 2, 3, 4], 'param_name': 'interpolation', 'type': 'radio'}], 'RandomRotate90': [], 'RandomScale': [{'defaults': [-0.1, 0.1], 'limits_list': [-0.9, 2.0], 'param_name': 'scale_limit', 'type': 'num_interval'}, {'options_list': [0, 1, 2, 3, 4], 'param_name': 'interpolation', 'type': 'radio'}], 'RandomSizedCrop': [{'param_name': 'min_max_height', 'type': 'num_interval', 'defaults': [106, 212], 'limits_list': [1, 212]}, {'param_name': 'height', 'type': 'num_interval', 'defaults': 212, 'limits_list': [1, 212]}, {'param_name': 'width', 'type': 'num_interval', 'defaults': 320, 'limits_list': [1, 320]}, {'defaults': 1.0, 'limits_list': [0.1, 1.0], 'param_name': 'w2h_ratio', 'type': 'num_interval'}, {'options_list': [0, 1, 2, 3, 4], 'param_name': 'interpolation', 'type': 'radio'}], 'RandomSnow': [{'defaults_list': [0.1, 0.2], 'limits_list': [0.0, 1.0], 'param_name': ['snow_point_lower', 'snow_point_upper'], 'type': 'min_max'}, {'defaults': 2.5, 'limits_list': [0.0, 5.0], 'param_name': 'brightness_coeff', 'type': 'num_interval'}], 'Resize': [{'param_name': 'height', 'type': 'num_interval', 'defaults': 106, 'limits_list': [1, 212]}, {'param_name': 'width', 'type': 'num_interval', 'defaults': 160, 'limits_list': [1, 320]}, {'options_list': [0, 1, 2, 3, 4], 'param_name': 'interpolation', 'type': 'radio'}], 'Rotate': [{'defaults': [-90, 90], 'limits_list': [-360, 360], 'param_name': 'limit', 'type': 'num_interval'}, {'options_list': [0, 1, 2, 3, 4], 'param_name': 'interpolation', 'type': 'radio'}, {'options_list': [0, 1, 2, 3, 4], 'param_name': 'border_mode', 'type': 'radio'}, {'param_name': 'value', 'type': 'rgb'}], 'ShiftScaleRotate': [{'defaults': [-0.06, 0.06], 'limits_list': [-1.0, 1.0], 'param_name': 'shift_limit', 'type': 'num_interval'}, {'defaults': [-0.1, 0.1], 'limits_list': [-2.0, 2.0], 'param_name': 'scale_limit', 'type': 'num_interval'}, {'defaults': [-90, 90], 'limits_list': [-360, 360], 'param_name': 'rotate_limit', 'type': 'num_interval'}, {'options_list': [0, 1, 2, 3, 4], 'param_name': 'interpolation', 'type': 'radio'}, {'options_list': [0, 1, 2, 3, 4], 'param_name': 'border_mode', 'type': 'radio'}, {'param_name': 'value', 'type': 'rgb'}], 'SmallestMaxSize': [{'defaults': 512, 'limits_list': [1, 1024], 'param_name': 'max_size', 'type': 'num_interval'}, {'options_list': [0, 1, 2, 3, 4], 'param_name': 'interpolation', 'type': 'radio'}], 'Solarize': [{'defaults': 128, 'limits_list': [0, 255], 'param_name': 'threshold', 'type': 'num_interval'}], 'ToGray': [], 'ToSepia': [], 'Transpose': [], 'VerticalFlip': []}

@st.cache
def get_images_list(path_to_folder: str) -> list:
    """Return the list of images from folder
    Args:
        path_to_folder (str): absolute or relative path to the folder with images
    """
    image_names_list = [
        x for x in os.listdir(path_to_folder) if x[-3:] in ["jpg", "peg", "png"]
    ]
    return image_names_list


@st.cache
def load_image(image_name: str, path_to_folder: str, bgr2rgb: bool = True):
    """Load the image
    Args:
        image_name (str): name of the image
        path_to_folder (str): path to the folder with image
        bgr2rgb (bool): converts BGR image to RGB if True
    """
    path_to_image = os.path.join(path_to_folder, image_name)
    image = cv2.imread(path_to_image)
    if bgr2rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def upload_image(bgr2rgb: bool = True):
    """Uoload the image
    Args:
        bgr2rgb (bool): converts BGR image to RGB if True
    """
    file = st.sidebar.file_uploader(
        "Upload your image (jpg, jpeg, or png)", ["jpg", "jpeg", "png"]
    )
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), 1)
    if bgr2rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def load_augmentations_config(
    placeholder_params: dict, path_to_config: str = "configs/augmentations.json"
) -> dict:
    """Load the json config with params of all transforms
    Args:
        placeholder_params (dict): dict with values of placeholders
        path_to_config (str): path to the json config file
    """
    with open(path_to_config, "r") as config_file:
        augmentations = json.load(config_file)
    for name, params in augmentations.items():
        params = [fill_placeholders(param, placeholder_params) for param in params]
    return augmentations

def fill_placeholders(params: dict, placeholder_params: dict) -> dict:
    """Fill the placeholder values in the config file
    Args:
        params (dict): original params dict with placeholders
        placeholder_params (dict): dict with values of placeholders
    """
    # TODO: refactor
    if "placeholder" in params:
        placeholder_dict = params["placeholder"]
        for k, v in placeholder_dict.items():
            if isinstance(v, list):
                params[k] = []
                for element in v:
                    if element in placeholder_params:
                        params[k].append(placeholder_params[element])
                    else:
                        params[k].append(element)
            else:
                if v in placeholder_params:
                    params[k] = placeholder_params[v]
                else:
                    params[k] = v
        params.pop("placeholder")
    return params


def get_placeholder_params(image):
    return {
        "image_width": image.shape[1],
        "image_height": image.shape[0],
        "image_half_width": int(image.shape[1] / 2),
        "image_half_height": int(image.shape[0] / 2),
    }


def select_image(path_to_images: str, interface_type: str = "Simple"):
    """ Show interface to choose the image, and load it
    Args:
        path_to_images (dict): path ot folder with images
        interface_type (dict): mode of the interface used
    Returns:
        (status, image)
        status (int):
            0 - if everything is ok
            1 - if there is error during loading of image file
            2 - if user hasn't uploaded photo yet
    """
    image_names_list = get_images_list(path_to_images)
    if len(image_names_list) < 1:
        return 1, 0
    else:
        if interface_type == "Professional":
            image_name = st.sidebar.selectbox(
                "Select an image:", image_names_list + ["Upload my image"]
            )
        else:
            image_name = st.sidebar.selectbox("Select an image:", image_names_list)

        if image_name != "Upload my image":
            try:
                image = load_image(image_name, path_to_images)
                return 0, image
            except cv2.error:
                return 1, 0
        else:
            try:
                image = upload_image()
                return 0, image
            except cv2.error:
                return 1, 0
            except AttributeError:
                return 2, 0


path_to_images = "/home/pasonatech/workspace/albumentations_forked/albumentations-demo/images/"
interface_type = "Professional"

status, image = select_image(path_to_images, interface_type)
# image was loaded successfully
placeholder_params = get_placeholder_params(image)
print(placeholder_params)
# load the config
augmentations = load_augmentations_config(
    placeholder_params, "/home/pasonatech/workspace/albumentations_forked/albumentations-demo/configs/augmentations.json"
)

transformation_list = [
    [
        "OneOf",
        [
            "Downscale",
            "Blur"
            ]
    ],
    [
        "OneOf",
        [
            "VerticalFlip",
            "HorizontalFlip"
            ]
    ],
    "GaussNoise"
]

# print(augmentations)
print(transformation_list)