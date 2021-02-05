import os
from albumentations.augmentations.bbox_utils import check_bbox
import streamlit as st
import albumentations as A
from streamlit.UploadedFileManager import File
import json
import time
from io import StringIO
import base64

from utils import (
    load_augmentations_config,
    get_arguments,
    get_placeholder_params,
    select_transformations,
    show_random_params,
    onetine_data_loader,
      
)
from visuals import (
    select_image,
    show_credentials,
    show_docstring,
    get_transormations_params,
    get_transormations_params_custom,
)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def main():
    local_css("src/custom_css.css")
    # logo_img = "/home/pasonatech/workspace/albumentations_forked/albumentations-demo/images/p.png"
    # html_sticky = f"""
    #     <div class="sticky pt-2">
    #         <img class="img-fluid" src="data:image/png;base64,{base64.b64encode(open(logo_img, "rb").read()).decode()}">
    #     </div>
    # """
    # st.markdown(html_sticky ,unsafe_allow_html = True)

    # get CLI params: the path to images and image width
    path_to_images, width_original = get_arguments()

    if not os.path.isdir(path_to_images):
        st.title("There is no directory: " + path_to_images)
    else:
        # select interface type
        interface_type = st.sidebar.radio(
            "Select the interface mode", ["Simple", "Professional", "Custom", "LoadMyFile"]
        )

        #pick css
        if interface_type == "LoadMyFile":
            local_css("src/custom_loadmy_css.css")









        if interface_type == "Custom":
            json_file_name = st.sidebar.text_input("Insert Json File Name", "aug_file")  #text_area same format
            json_file_name=os.path.join("./my_json_files",f"{json_file_name}"+'.json')                       
        
        # select image
        status, image = select_image(path_to_images, interface_type)
        if status == 1:
            st.title("Can't load image")
        if status == 2:
            st.title("Please, upload the image")
        else:
            # image was loaded successfully
            placeholder_params = get_placeholder_params(image)

            # load the config
            augmentations = load_augmentations_config(
                placeholder_params, "configs/augmentations.json"
            )

            if interface_type is not "LoadMyFile":
                # get the list of transformations names
                transform_names = select_transformations(augmentations, interface_type)
            
            if interface_type is "Custom":
                transforms = get_transormations_params_custom(transform_names, augmentations,json_file_name)
            
            elif interface_type is "LoadMyFile":  
                
                


                f_name = st.sidebar.file_uploader("Select your json file",type = "json")                 
                
                view_times=0               
                if f_name:
                    j_text = StringIO.read(f_name)
                    j_data = json.loads(j_text)
                    
                    image_replace = st.empty()
                    st.image(image, caption="Original image", width=width_original)
                    if st.sidebar.button("Play Preview"):
                        view_times = 1
                    stop_btn = st.sidebar.button("STOP Preview")
                    if stop_btn:
                        view_times = 0
                    # for seconds in range(view_times):
                    # data =j_data 
                    try:
                        transform = A.from_dict(j_data)
                        display_value = True
                    except KeyError:
                        st.error("Please, confirm your augmentations structure.")
                        st.error("Supports only albumentations augmentation generated 'A.to_dict()'.")
                        # view_times = 0
                        display_value = False
                                       
                    while(view_times == 1):
                        
                        try:
                            # data = json.load(open(file_name, 'r'))                                               
                            # transform = A.from_dict(data)
                            aug_img_obj = transform(image=image)
                            # print(aug_img_obj.keys())
                            aug_img = aug_img_obj['image']
                            
                            image_replace.image(
                                aug_img,
                                caption="Transformed image",
                                width=width_original,
                            ) 
                        except IOError:
                            st.error("Confirm your json file path.")
                            view_times = 0
                        except UnboundLocalError:
                            st.error("Your json file seems incompatible to run this task. ")
                            view_times = 0
                        except ValueError as e:
                            image_replace.error(e)  #replaces error log in same field
                            pass
                        
                        time.sleep(1)
                    if stop_btn is True:
                        st.info("Preview Stopped. Press Play Preview button to resume previewing.")
                    if display_value:
                        if st.sidebar.checkbox("Display Augmentation Parameters"):
                            onetine_data_loader(j_data)
                                                
                    transforms =[]
                else:
                    st.header("WELCOME")
                    st.header("Please upload a JSON File")


            else:
                # get parameters for each transform
                transforms = get_transormations_params(transform_names, augmentations)

            
            if interface_type is not "LoadMyFile":
                try:
                    # apply the transformation to the image
                    data = A.ReplayCompose(transforms)(image=image)
                    error = 0
                except ValueError:
                    error = 1
                    st.title(
                        "The error has occurred. Most probably you have passed wrong set of parameters. \
                    Check transforms that change the shape of image."
                    )

                # proceed only if everything is ok
                if error == 0:
                    augmented_image = data["image"]
                    # show title
                    st.title("Demo of Albumentations")

                    # show the images
                    width_transformed = int(
                        width_original / image.shape[1] * augmented_image.shape[1]
                    )

                    st.image(image, caption="Original image", width=width_original)
                    st.image(
                        augmented_image,
                        caption="Transformed image",
                        width=width_transformed,
                    )

                    # comment about refreshing
                    st.write("*Press 'R' to refresh*")

                    #custom preview of aug list
                    # random values used to get transformations
                    show_random_params(data, interface_type)

                    for transform in transforms:
                        show_docstring(transform)
                        st.code(str(transform))
                    show_credentials()

                # adding google analytics pixel
                # only when deployed online. don't collect statistics of local usage
                if "GA" in os.environ:
                    st.image(os.environ["GA"])
                    st.markdown(
                        (
                            "[Privacy policy]"
                            + (
                                "(https://htmlpreview.github.io/?"
                                + "https://github.com/IliaLarchenko/"
                                + "albumentations-demo/blob/deploy/docs/privacy.html)"
                            )
                        )
                    )


if __name__ == "__main__":
    main()
