import streamlit as st
from PIL import Image
import cv2
import os
from SOURCE.yolo_files import detect
from SOURCE.gan_files import test
from SOURCE.vgg_finetuned_model import vgg_verify
from helper_fns import gan_utils
import shutil
import glob
from SessionState import get_session
import time

MEDIA_ROOT = 'media/documents/'
SIGNATURE_ROOT = 'media/UserSignaturesSquare/'
YOLO_RESULT = 'results/yolov5/'
YOLO_OP = 'crops/DLSignature/'
GAN_IPS = 'results/gan/gan_signdata_kaggle/gan_ips/testB'
GAN_OP = 'results/gan/gan_signdata_kaggle/test_latest/images/'
GAN_OP_RESIZED = 'results/gan/gan_signdata_kaggle/test_latest/images/'

def select_cleaned_image(selection):
    ''' Returns the path of cleaned image corresponding to the document the user selected '''
    base_filename = os.path.splitext(os.path.basename(selection))[0]
    cleaned_image_path = os.path.join(GAN_OP, f"{base_filename}_fake.png")
    cleaned_image_path = cleaned_image_path.replace('\\', '/')
    return cleaned_image_path

def copy_and_overwrite(from_path, to_path):
    ''' Copy files from results/yolo_ops/ to results/gan/gan_signdata_kaggle/gan_ips '''
    if os.path.exists(to_path):
        shutil.rmtree(to_path)
    shutil.copytree(from_path, to_path)

def signature_verify(selection):
    ''' Performs signature verification '''
    base_filename = os.path.splitext(os.path.basename(selection))[0]
    anchor_image = os.path.join(SIGNATURE_ROOT, f"{base_filename}.png")
    anchor_image = anchor_image.replace('\\', '/')
    if not os.path.exists(anchor_image):
        st.error(f"Anchor image not found at: {anchor_image}")
        return
    feature_set = vgg_verify.verify(anchor_image, GAN_OP_RESIZED)
    if feature_set:  # make sure there's at least one result
        image, score = feature_set[0]

        columns = st.columns(3)

        # Display Original Signature
        with columns[0]:
            st.image(anchor_image, caption="Original Signature", use_container_width=True)

        # Display Detected Signature
        with columns[1]:
            st.image(image, caption="Detected Signature", use_container_width=True)

        # Display Score with Color and Style
        with columns[2]:
            st.subheader("Result:")
            st.write(f"Similarity Score: {score * 100:.0f}%")
            if score >= 0.80:
                st.success('The Signature is Genuine')
            else:
                st.error('The Signature is Forged')


def signature_cleaning(selection, yolo_op):
    ''' Performs signature cleaning and displays the cleaned signatures '''
    copy_and_overwrite(yolo_op, GAN_IPS)
    test.clean()  # performs cleaning
    cleaned_image = select_cleaned_image(selection)
    if not os.path.exists(cleaned_image):
        st.error(f"Cleaned image not found at {cleaned_image}")
        return
    st.image(cleaned_image)

def signature_detection(document_path):
    ''' Performs signature detection on the uploaded document '''
    detect.detect(document_path)
    latest_detection = max(glob.glob(os.path.join(YOLO_RESULT, '*/')), key=os.path.getmtime)
    gan_utils.resize_images(os.path.join(latest_detection, YOLO_OP))

    base_filename = os.path.splitext(os.path.basename(document_path))[0]
    selection_detection = os.path.join(latest_detection, YOLO_OP, base_filename + '.jpg')
    selection_detection = selection_detection.replace('\\', '/')

    for _ in range(5):
        if os.path.exists(selection_detection):
            st.image(selection_detection)
            break
        else:
            time.sleep(1)

    if not os.path.exists(selection_detection):
        st.error(f"Error: Could not find the detected signature file: {selection_detection}")

    return os.path.join(latest_detection, YOLO_OP)

def select_document():
    ''' Allows the user to upload a document, saves it to the media/documents directory '''
    st.header("Upload a Document")
    uploaded_file = st.file_uploader("Upload the document image (e.g., png, jpg, jpeg)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        save_path = os.path.join(MEDIA_ROOT, uploaded_file.name)

        if not os.path.exists(MEDIA_ROOT):
            os.makedirs(MEDIA_ROOT)

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(save_path, caption="Uploaded Document", use_container_width=True)
        return save_path
    else:
        st.warning("Please upload a document to proceed.")
        return None

def select_signature():
    ''' Allows the user to select an existing signature from SIGNATURE_ROOT '''
    st.header("Select an Original Signature")
    
    # Create two columns
    col1, col2 = st.columns(2)

    # Display select box on the left column
    with col1:
        signature_files = [f for f in os.listdir(SIGNATURE_ROOT) if os.path.isfile(os.path.join(SIGNATURE_ROOT, f))]
        selected_signature = st.selectbox("Select a signature file:", signature_files if signature_files else ["No signatures available"])

    # Display image on the right column
    if selected_signature and selected_signature != "No signatures available":
        signature_path = os.path.join(SIGNATURE_ROOT, selected_signature)
        with col2:
            st.image(signature_path, caption="Selected Signature", use_container_width=True)
        return signature_path
    else:
        st.warning("Please select a valid signature to proceed.")
        return None

def main():
    session_state = get_session(
        signature_selection='',
        document_selection='',
        yolo_op='',
        detect_button=False,
        clean_button=False,
        verify_button=False,
    )


    st.markdown("""
        <div style="border: 2px solid #808080; padding: 10px; border-radius: 5px; margin-bottom: 60px; background-color: #262730; border-radius: 10px;">
            <h4>An Enhancement of Harris Corner Detector Algorithm for Signature Forgery Detection System</h4>
            <h6>Authors: Dzelle Faith Tan, Pauline Regina Obispo, Jonathan C. Morano, Khatalyn E. Mata</h6>
            <p>Computer Science Department, College of Information Systems and Technology Management (CISTM), Pamantasan ng Lungsod ng Maynila</p>
        </div>
    """, unsafe_allow_html=True)


    # Step 1: Select an existing signature
    session_state.signature_selection = select_signature()

    # Step 2: Select a document if a signature has been selected
    if session_state.signature_selection:
        session_state.document_selection = select_document()

        # Proceed only if both signature and document have been selected
        if session_state.signature_selection and session_state.document_selection:
            detect_button = st.button('Detect Signature')
            if detect_button:
                session_state.detect_button = True
            if session_state.detect_button:
                session_state.yolo_op = signature_detection(session_state.document_selection)

                clean_button = st.button('Clean Signature')
                if clean_button:
                    session_state.clean_button = True
                if session_state.clean_button:
                    signature_cleaning(session_state.document_selection, session_state.yolo_op)

                    verify_button = st.button('Verify Signature')
                    if verify_button:
                        session_state.verify_button = True
                    if session_state.verify_button:
                        signature_verify(session_state.signature_selection)

                        # Cleanup the directories at the end
                        if os.path.exists('results'):
                            shutil.rmtree('results')
                        if os.path.exists(MEDIA_ROOT):
                            shutil.rmtree(MEDIA_ROOT)
                        # for item in os.listdir(SIGNATURE_ROOT):
                        #     item_path = os.path.join(SIGNATURE_ROOT, item)
                        #     if os.path.isfile(item_path) or os.path.islink(item_path):
                        #         os.unlink(item_path)
                        #     elif os.path.isdir(item_path):
                        #         shutil.rmtree(item_path)

if __name__ == "__main__":
    main()