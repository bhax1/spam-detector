import streamlit as st
import re
from joblib import load
from streamlit_extras.colored_header import colored_header
from streamlit_extras.let_it_rain import rain

# Load the trained model and vectorizer
@st.cache_resource
def load_model():
    model = load("SpamDetector.pkl")
    vectorizer = load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# Text cleaning function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Streamlit UI configuration
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📱",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stTextArea textarea {
            border: 2px solid #dee2e6;
            border-radius: 10px;
            padding: 10px;
        }
        .stButton>button {
            background-color: #4a6bdf;
            color: white;
            border-radius: 8px;
            padding: 10px 24px;
            font-weight: bold;
            border: none;
            width: 100%;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #3a56c0;
            transform: scale(1.02);
        }
        .success-box {
            background-color: #d4edda;
            color: #155724;
            padding: 15px;
            border-radius: 10px;
            border-left: 5px solid #28a745;
            margin: 10px 0;
        }
        .error-box {
            background-color: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 10px;
            border-left: 5px solid #dc3545;
            margin: 10px 0;
        }
        .info-box {
            background-color: #e7f5ff;
            color: #0c5460;
            padding: 15px;
            border-radius: 10px;
            border-left: 5px solid #17a2b8;
            margin: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

# Header section
colored_header(
    label="📱 SMS Spam Detector",
    description="Check if your message is spam or ham (not spam)",
    color_name="blue-70",
)

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.write("""
        This app uses machine learning to classify SMS messages as spam or not spam (ham).
        The model was trained on a dataset of labeled SMS messages.
        
        **How it works:**
        1. Enter your message in the text box
        2. Click the "Detect" button
        3. Get instant classification results
    """)
    
    st.divider()
    
    st.subheader("Examples to try:")
    st.markdown("""
        - **Spam:** "WINNER!! You've been selected for a free $1000 gift card!"
        - **Ham:** "Hey, are we still meeting for lunch tomorrow?"
        - **Spam:** "Urgent: Your bank account needs verification. Click here now!"
    """)

# Main content
st.subheader("Enter your SMS message below")

user_input = st.text_area(
    "Message content:",
    height=150,
    placeholder="Type or paste your SMS message here...",
    label_visibility="collapsed"
)

col1, col2 = st.columns([1, 1])
with col1:
    detect_btn = st.button("Detect", type="primary", use_container_width=True)
with col2:
    clear_btn = st.button("Clear", use_container_width=True)

if clear_btn:
    user_input = ""

if detect_btn:
    if not user_input.strip():
        st.warning("Please enter a message before detection.")
    else:
        with st.spinner("Analyzing message..."):
            # Clean and vectorize the input
            cleaned_text = preprocess(user_input)
            vectorized_text = vectorizer.transform([cleaned_text])

            # Predict using the loaded model
            prediction = model.predict(vectorized_text)[0]
            prediction_proba = model.predict_proba(vectorized_text)[0]
            
            # Get confidence percentage
            confidence = max(prediction_proba) * 100
            
            # Display results
            if prediction == "spam" or prediction == 1:
                rain(
                    emoji="🚫",
                    font_size=30,
                    falling_speed=5,
                    animation_length=1,
                )
                st.markdown(f"""
                    <div class="error-box">
                        <h3>🚫 SPAM DETECTED</h3>
                        <p>This message is classified as <strong>SPAM</strong> with {confidence:.1f}% confidence.</p>
                        <p><em>Tip: Be cautious with messages asking for personal information or offering unexpected prizes.</em></p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="success-box">
                        <h3>✅ NOT SPAM</h3>
                        <p>This message is classified as <strong>HAM</strong> (not spam) with {confidence:.1f}% confidence.</p>
                        <p><em>Note: Always use your best judgment when receiving messages.</em></p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Show additional information
            with st.expander("Details"):
                st.write(f"**Original message:** {user_input}")
                st.write(f"**Processed text:** {cleaned_text}")
                st.write(f"**Prediction probabilities:**")
                st.write(f"- Spam: {prediction_proba[1]*100:.1f}%")
                st.write(f"- Ham: {prediction_proba[0]*100:.1f}%")

# Footer
st.divider()
st.caption("""
    This is a machine learning model and may not be 100% accurate. 
    Always verify suspicious messages through official channels.
""")