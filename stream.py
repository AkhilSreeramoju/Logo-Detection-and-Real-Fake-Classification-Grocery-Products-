import streamlit as st
from logo import predict
from PIL import Image

st.title('Real or Fake Logo Detection 🛒')

uploaded = st.file_uploader('Upload image', type=['jpg', 'png', 'jpeg'])

if uploaded:
    image = 'temp.jpg'
    with open(image, 'wb') as f:
        f.write(uploaded.getbuffer())

    processed_img, predictions = predict(image)

    st.image(processed_img, caption='Uploaded Image', use_container_width=True)

    if predictions:
        st.success(f"Detected Logos: {', '.join(predictions)}")
    else:
        st.warning("⚠️ No logos detected in the uploaded image Fake.")