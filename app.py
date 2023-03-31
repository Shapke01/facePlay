import cv2
import streamlit as st
import numpy as np
from PIL import Image
from pyface import Face, put_face_on_image, combine_faces, face_morphing



st.set_page_config(layout="wide")
st.markdown(""" <style>
    footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)


new_faces = st.sidebar.file_uploader(label = "Add face image :blush: (preffered size: 600x400)",
                                accept_multiple_files = True)

faces = list()
names = list()

for face in new_faces:
    names.append(face.name.split('.')[0])
    faces.append(Face(cv2.resize(np.array(Image.open(face)), (600,400))))


left, _, right = st.columns([2,0.2,1])

with left:
    # TITLE
    st.markdown("<h1 style='text-align: center; color: white;'>Loaded Faces</h1>", unsafe_allow_html=True)
    st.markdown("""---""")
    cols = st.columns(max(len(faces),1))
    sliders = [None] * len(faces)
    checkboxes = np.array([None] * len(faces))
    if len(faces) > 0:
        for idx, (face, col) in enumerate(zip(faces, cols)):
            with col:
                st.markdown(f"<p style='text-align: center; color: white;'>{names[idx]}</p>", unsafe_allow_html=True)
                st.image(put_face_on_image(face, np.zeros((400, 600, 3), dtype=np.uint8)))
                sliders[idx] = st.slider(str(idx), 0., 1., 0.01, label_visibility='collapsed')
                checkboxes[idx] = st.checkbox(str(idx), label_visibility='collapsed')


with right:
    st.markdown("<h1 style='text-align: center; color: white;'>Results</h1>", unsafe_allow_html=True)
    st.markdown("""---""")

    if len(faces) > 1:
        if (np.array(sliders) > 0).any():
            combined = combine_faces(faces=faces, weights=sliders)
            img = put_face_on_image(combined, np.zeros((400, 600, 3), dtype=np.uint8))
            st.markdown(f"<p style='text-align: center; color: white;'>Combined</p>", unsafe_allow_html=True)
            st.image(img)

            
            
        if checkboxes.sum() == 2:
            if st.button('Combine checked faces into GIF!'):
                indices = np.where(checkboxes == True)[0]
                face_morphing([faces[indices[0]], faces[indices[1]]])




































# run = st.checkbox('Run')
# FRAME_WINDOW = st.image([])
# camera = cv2.VideoCapture(0)

# while run:
#     _, frame = camera.read()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    
#     FRAME_WINDOW.image(frame)
# else:
#     st.write('Stopped')