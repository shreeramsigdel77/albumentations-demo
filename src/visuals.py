import streamlit as st

from control import param2func
from utils import get_images_list, load_image


def show_logo():
    st.image(load_image("logo.png", "../images"), format="PNG")


def select_image(path_to_images: str = "images"):
    image_names_list = get_images_list(path_to_images)
    image_name = st.sidebar.selectbox("Select an image:", image_names_list)
    image = load_image(image_name, path_to_images)
    return image


def show_transform_control(transform_params: dict):
    param_values = {'p': 1.0}
    if len(transform_params) == 0:
        st.sidebar.text("Transform has no parameters")
    else:
        for param in transform_params:
            control_function = param2func[param["type"]]
            param_values[param["param_name"]] = control_function(**param)
    return param_values


def show_credentials():
    st.text("")
    st.text("")
    st.subheader("Credentials:")
    st.text("Source: github.com/IliaLarchenko/albumentations-demo")
    st.text("Albumentations library: github.com/albumentations-team/albumentations")
    st.text("Image Source: pexels.com/royalty-free-images/")


def show_docstring(obj_with_ds):
    st.subheader("Docstring:")
    st.text(str(obj_with_ds.__doc__))
