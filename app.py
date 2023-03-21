import cv2
import streamlit as st
import numpy as np
from PIL import Image
from pyface import Face, put_face_on_image, combine_faces, face_morphing



st.set_page_config(layout="wide")
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

# TITLE
st.markdown("<h1 style='text-align: center; color: white;'>FACE APP</h1>", unsafe_allow_html=True)
st.markdown("""---""")
new_faces = st.sidebar.file_uploader(label = "Add face image :blush: (preffered size: 600x400)",
                                accept_multiple_files = True)

faces = list()
names = list()

for face in new_faces:
    names.append(face.name.split('.')[0])
    faces.append(Face(cv2.resize(np.array(Image.open(face)), (600,400))))

cols = st.columns(max(len(faces),1))
sliders = [None] * len(faces)
if len(faces) > 0:
    for idx, (face, col) in enumerate(zip(faces, cols)):
        with col:
            st.markdown(f"<p style='text-align: center; color: white;'>{names[idx]}</p>", unsafe_allow_html=True)
            st.image(put_face_on_image(face, np.zeros((400, 600, 3), dtype=np.uint8)))
            sliders[idx] = st.slider(str(idx), 0., 1., 0.01, label_visibility='collapsed')

print(sliders)

st.markdown("""---""")
st.markdown("<h2 style='text-align: center; color: white;'>COMBINED</h2>", unsafe_allow_html=True)

combined = combine_faces(faces=faces, weights=sliders)
img = put_face_on_image(combined, np.zeros((400, 600, 3), dtype=np.uint8))
_, col, _ = st.columns([1,2,1])
with col:
    st.image(img)



































# run = st.checkbox('Run')
# FRAME_WINDOW = st.image([])
# camera = cv2.VideoCapture(0)

# while run:
#     _, frame = camera.read()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    
#     FRAME_WINDOW.image(frame)
# else:
#     st.write('Stopped')