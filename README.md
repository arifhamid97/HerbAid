# HerbAid

HerbAid is a medicinal plant leaf classification app designed to help users identify plants based on leaf images and provide information on their uses and how to use them. 

🔧 **Technologies Used:**
- Fine-tuned Vision Transformer model from Hugging Face
- OpenAI GPT-3.5 Turbo
- LangChain
- Transformers

🔍 **Project Flow:**
1. **Plant Identification:**
   - The fine-tuned Vision Transformer model from Hugging Face processes the image and identifies the plant species.
   - This can be easily done by creating a Hugging Face pipeline for the vision model.
2. **Generating and Extracting Plant Usage Information:**
   - The identified plant species is used to prompt OpenAI's GPT-3.5 Turbo via LangChain to generate detailed information about the plant’s medicinal uses and instructions on how to use it.
   - By utilizing Pydantic models, the output from GPT-3.5 Turbo is structured and extracted into a dictionary format for easy use.

3. **User Interface:**
   - The application is bootstrapped using Streamlit, providing an intuitive and interactive user interface.
   - Users can easily upload images, view plant identification results, and read detailed information about the plant's medicinal properties and usage.

⚠️ **Remarks:**
   - The Vision Transformer model is fine-tuned for 40 different leaves. As a result, the accuracy might not be perfect for all plant species.
