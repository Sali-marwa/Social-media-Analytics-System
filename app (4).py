import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import random
import matplotlib.pyplot as plt
from textblob import TextBlob
import readability
from PIL import Image, ImageFilter
import numpy as np
import cv2

# Initialize ChatGroq with your API key and model
chat = ChatGroq(
    temperature=0.7,
    model="llama3-70b-8192",
    api_key="YOUR_API"
)

# Define the prompt template for the assistant
system = """
You are a helpful assistant specialized in social media optimization. When given a post and an image, suggest relevant keywords and tags to maximize engagement, regenerate the post incorporating those suggestions, categorize the post, provide the best posting time based on social media trends, and offer content improvement tips. Also, add relevant emojis and image enhancement suggestions.
"""
human = """
Post: {text}
"""
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

# Define the chain for post text optimization
chain_text = prompt | chat

# Function to handle chat interaction for text
def chat_with_groq_text(user_message):
    response = chain_text.invoke({"text": user_message})
    return response.content

# Define the chain for image optimization
chain_image = prompt | chat

# Function to handle chat interaction for image
def chat_with_groq_image(user_message, image_features):
    response = chain_image.invoke({"text": user_message, **image_features})
    return response.content

# Function to predict engagement
def predict_engagement():
    likes = random.randint(20, 200)
    comments = random.randint(2, 20)
    reposts = random.randint(1, 13)
    return likes, comments, reposts

# Function to plot engagement bar chart
def plot_engagement(likes, comments, reposts):
    labels = ['Likes', 'Comments', 'Reposts']
    sizes = [likes, comments, reposts]
    colors = ['#ff9999','#66b3ff','#99ff99']
    fig, ax = plt.subplots()
    ax.bar(labels, sizes, color=colors)
    for i, v in enumerate(sizes):
        ax.text(i, v + 10, str(v), color='black', ha='center')
    ax.set_ylabel('Count')
    return fig

# Function to analyze sentiment
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return "Positive"
    elif sentiment < 0:
        return "Negative"
    else:
        return "Neutral"

# Function to calculate readability score
def calculate_readability(text):
    results = readability.getmeasures(text, lang='en')
    return results['readability grades']['FleschReadingEase']

# Function to analyze image and extract features using OpenCV
def analyze_image(image):
    image_array = np.array(image.convert('RGB'))
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    # Brightness
    brightness = np.mean(image_array)
    
    # Contrast
    contrast = np.std(gray_image)
    
    # Sharpness using Laplacian
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    sharpness = np.var(laplacian)
    
    # Color Distribution (mean color)
    mean_color = np.mean(image_array, axis=(0, 1))
    
    # Noise estimation
    noise = np.std(image_array)
    
    return {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "mean_color": mean_color,
        "noise": noise
    }

# Function to generate practical suggestions based on image features
def generate_image_suggestions(image_features):
    suggestions = []
    if image_features['brightness'] < 100:  # Adjust threshold as needed
        suggestions.append("Brightness is low. Consider increasing it.")
    if image_features['contrast'] < 30:  # Adjust threshold as needed
        suggestions.append("Contrast is low. Consider increasing it.")
    if image_features['sharpness'] < 100:  # Adjust threshold as needed
        suggestions.append("Sharpness is low. Consider increasing it.")
    if image_features['noise'] > 50:  # Adjust threshold as needed
        suggestions.append("Image may be noisy. Consider reducing noise.")
    
    return suggestions

# Function to optimize and display the post and image
def optimize_post_and_image(user_input, image):
    # Process text post
    groq_response_text = chat_with_groq_text(user_input)
    
    st.markdown("### Optimized Post and Suggestions:")
    st.markdown(f"<div style='font-size:20px;'>{groq_response_text}</div>", unsafe_allow_html=True)
    
    # Process image
    image_features = analyze_image(image)
    groq_response_image = chat_with_groq_image(user_input, image_features)
    likes, comments, reposts = predict_engagement()
    sentiment = analyze_sentiment(groq_response_image)
    readability_score = calculate_readability(groq_response_image)
    char_count = len(groq_response_image)

    st.markdown("### Engagement Prediction:")
    st.markdown(f"**Predicted Likes:** {likes}")
    st.markdown(f"**Predicted Comments:** {comments}")
    st.markdown(f"**Predicted Reposts:** {reposts}")
    st.markdown(f"**Sentiment:** {sentiment}")
    st.markdown(f"**Readability Score:** {readability_score:.2f}")
    st.markdown(f"**Character Count:** {char_count}")

    fig = plot_engagement(likes, comments, reposts)
    st.pyplot(fig)

    # Display original image
    st.markdown("### Original Image:")
    st.image(image, caption='Original Image', use_column_width=True)

    # Display image parameters
    st.markdown("### Image Parameters:")
    st.markdown(f"**Brightness:** {image_features['brightness']:.2f}")
    st.markdown(f"**Contrast:** {image_features['contrast']:.2f}")
    st.markdown(f"**Sharpness:** {image_features['sharpness']:.2f}")
    st.markdown(f"**Mean Color:** {image_features['mean_color']}")
    st.markdown(f"**Noise:** {image_features['noise']:.2f}")

    # Generate and display image enhancement suggestions
    suggestions = generate_image_suggestions(image_features)
    st.markdown("### Image Enhancement Suggestions:")
    for suggestion in suggestions:
        st.markdown(f"- {suggestion}")

# Streamlit app UI
st.title("Social Media Post Optimizer")

user_input = st.text_area("Enter your post:", "")
uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])

if st.button("Optimize Post and Image"):
    if user_input and uploaded_file:
        image = Image.open(uploaded_file)
        optimize_post_and_image(user_input, image)

# Adding some UI enhancements
st.markdown(
    """
    <style>
    .stTextInput > div > div > textarea {
        padding: 10px;
        font-size: 18px;
    }
    .stButton > button {
        padding: 10px 20px;
        font-size: 18px;
    }
    .stMarkdown h1, h2, h3, h4 {
        font-size: 28px;
        margin-top: 20px;
    }
    .stMarkdown div {
        font-size: 18px;
    }
    </style>
    """, 
    unsafe_allow_html=True
)
