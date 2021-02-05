from json import dump
import cv2
from cv2 import data
import streamlit as st

import albumentations as A

from control import param2func
from utils import get_images_list, load_image, upload_image


def show_logo():
    st.image(load_image("logo.png", "../images"), format="PNG")


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
        elif interface_type == "Custom":
            image_name = st.sidebar.selectbox(
                "Select an image:", image_names_list + ["Upload my image"]
            )

        elif interface_type == "LoadMyFile":
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


def show_transform_control(transform_params: dict, n_for_hash: int) -> dict:
    #transform_params = augmentation["blur"]
    # [{'defaults': [3, 7], 'limits_list': [3, 100], 'param_name': 'blur_limit', 'type': 'num_interval'}]
    param_values = {"p": 1.0}
    if len(transform_params) == 0:
        st.sidebar.text("Transform has no parameters")
    else:
        for param in transform_params:
            control_function = param2func[param["type"]]
            if isinstance(param["param_name"], list):
                returned_values = control_function(**param, n_for_hash=n_for_hash)
                for name, value in zip(param["param_name"], returned_values):
                    param_values[name] = value
            else:
                param_values[param["param_name"]] = control_function(
                    **param, n_for_hash=n_for_hash
                )
        # st.write(param_values)
    return param_values


def show_credentials():
    st.markdown("* * *")
    st.subheader("Credentials:")
    st.markdown(
        (
            "Source: [github.com/IliaLarchenko/albumentations-demo]"
            "(https://github.com/IliaLarchenko/albumentations-demo)"
        )
    )
    st.markdown(
        (
            "Albumentations library: [github.com/albumentations-team/albumentations]"
            "(https://github.com/albumentations-team/albumentations)"
        )
    )
    st.markdown(
        (
            "Image Source: [pexels.com/royalty-free-images]"
            "(https://pexels.com/royalty-free-images/)"
        )
    )


def get_transormations_params(transform_names: list, augmentations: dict) -> list:
    transforms = []
    for i, transform_name in enumerate(transform_names):
        # select the params values
        st.sidebar.subheader("Params of the " + transform_name)
        param_values = show_transform_control(augmentations[transform_name], i)
        transforms.append(getattr(A, transform_name)(**param_values))
    return transforms

#change log
def get_transormations_params_custom(transform_names: list, augmentations: dict,json_fil_name:str) -> list:
    transforms = []
    temp_names = []
    test_transforms = []  
    my_test_dict_temp = []
    for item in transform_names:
        temp_transforms = []
        my_test_dict_temp1 = []
        
        if isinstance(item,list):
            #ToDO 
            for i, transform_name in enumerate(item[1]):
                # select the params values
                st.sidebar.subheader("Params of the " + transform_name)
                param_values = show_transform_control(augmentations[transform_name], i)
                
                #new
                my_test_dict = A.to_dict(getattr(A,str(transform_name))(**param_values))
                my_test_dict['transform']['__class_fullname__'] = my_test_dict['transform']['__class_fullname__'].split('.')[-1]
                my_test_dict_temp1.append(my_test_dict['transform'])

                transforms.append(getattr(A, transform_name)(**param_values))
                temp_transforms.append(getattr(A, transform_name)(**param_values))

            
            
            my_test_dict_temp2 = {
                '__class_fullname__': item[0], 
                'p': 0.5,
                'transforms':my_test_dict_temp1,
            }
            if item[0] == "SomeOf":
                add_dict = {'sample_range':[2,4]}
                my_test_dict_temp2.update(add_dict)
            my_test_dict_temp.append(my_test_dict_temp2)
            test_transforms.append([item[0],temp_transforms])
        
        else:
            temp_names.append(item)
       
    if temp_names is not None:
        for i, transform_name in enumerate(temp_names):
            
            # select the params values
            st.sidebar.subheader("Params of the " + transform_name)
            param_values = show_transform_control(augmentations[transform_name], i)

            #get dict format
            my_test_dict = A.to_dict(getattr(A,str(transform_name))(**param_values))
            my_test_dict['transform']['__class_fullname__'] = my_test_dict['transform']['__class_fullname__'].split('.')[-1]
            my_test_dict_temp.append(my_test_dict['transform'])
            # my_test_dict_temp1.append(my_test_dict['transform'])
            
            # my_test_dict_temp = {
            #     '__version__': '0.5.2',
            #     'transform': {
            #         '__class_fullname__': 'Compose', 'p': 1.0,
            #         # 'transforms':  my_test_dict_temp
            #         'transforms':  my_test_dict_temp1
            #     }
            # }          
            
            transforms.append(getattr(A, transform_name)(**param_values))
            test_transforms.append(getattr(A,transform_name)(**param_values))
        
        my_test_dict_temp = {
                '__version__': '0.5.2',
                'transform': {
                    '__class_fullname__': 'Compose', 'p': 1.0,
                    'transforms':  my_test_dict_temp
                }
            }
        if st.checkbox("Preview of Augmentation Parameters"):
            st.write(my_test_dict_temp)

    if st.sidebar.button("Save"):
        save_json_data(file_name= json_fil_name,dict= my_test_dict_temp)
        st.sidebar.success("File has been saved.")
    albu_preview(test_transforms)
    
    return transforms


def save_json_data(file_name:str,dict:dict):
    import json
    
    with open(file_name, 'w') as f:
        json.dump(dict,f,indent=4)
    f.close()

def albu_preview(test_transforms):
    
    for item in test_transforms:
        if isinstance(item,list):
            
            st.code(f"{item[0]}")
            temp_oneof =[]
            for aug_names in item[1]:
                st.code(f"      {aug_names}")
                temp_oneof.append(aug_names)      

        else:
            st.code(f"{item}")


def show_docstring(obj_with_ds):
    st.markdown("* * *")
    st.subheader("Docstring for " + obj_with_ds.__class__.__name__)
    st.text(obj_with_ds.__doc__)
