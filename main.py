import streamlit as st
from PIL import Image
from core import HerbAid

herbAid = HerbAid()
st.title("HerbAid")

st.write("Upload an image of a medicinal plant leave and get its usage and how-to-use information.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])

    with col1:
        # Display the uploaded image in a smaller size
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', width=150)
    
    with col2:
        st.write("")

        # Adding a loading spinner during image processing
        with st.spinner('Classifying...'):
            plant_details = herbAid.invoke_chain(image)

        if plant_details == None:
            st.markdown(f"### **The Image is not medicinal plant**")
        else:
            # Display the plant information
            st.markdown(f"### **Plant Name:** {plant_details['plant_name']}")
            st.markdown(f"### **Usage:**\n{plant_details['usage']}")
            st.markdown(f"### **How to Use:**\n{plant_details['how_to_use']}")
