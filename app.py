import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

st.set_page_config(layout="wide")
st.title("🧠 Facial Expression Monitoring System")

IMG_SIZE = (48, 48)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("expression_model.h5")

model = load_model()
labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def send_email_alert(expression, recipient_email):
    sender_email = "u2334716@gmail.com"
    sender_password = "escs rgmd jufz ovpl"  

    subject = "🚨 Patient Expression Alert"
    body = f"Alert: Patient is showing a '{expression.upper()}' expression. Please check immediately."

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()
        st.success(f"📧 Email alert sent to {recipient_email}")
    except Exception as e:
        st.warning(f"❌ Failed to send email: {e}")

def get_prediction_label(frame):
    resized = cv2.resize(frame, IMG_SIZE)
    input_array = np.expand_dims(resized / 255.0, axis=0)
    prediction = model.predict(input_array)[0]
    return labels[np.argmax(prediction)]

def show_alert(expression):
    st.markdown(f"<h1 style='text-align: center; color: darkblue;'>Detected Expression: {expression.upper()}</h1>", unsafe_allow_html=True)
    if expression in ["angry", "fear", "sad"]:
        st.error("🚨 EMERGENCY ALERT: Patient may be distressed. Notifying caretaker...")
        if recipient_email:
            send_email_alert(expression, recipient_email)
        else:
            st.warning("⚠️ No caretaker email address provided.")
    elif expression == "happy":
        st.success("😊 Patient appears happy and stable.")
    elif expression == "surprise":
        st.warning("⚠️ Unusual facial expression detected. Monitor closely.")
    elif expression == "neutral":
        st.info("😐 Patient expression neutral. Normal observation.")

st.sidebar.header("Notification Settings")
recipient_email = st.sidebar.text_input("Caretaker Email", placeholder="example@gmail.com")

st.header("📷 Upload Image")
uploaded_image = st.file_uploader("Upload a facial image", type=["jpg", "png", "jpeg"])
if uploaded_image:
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(img, caption="Uploaded Image", channels="BGR")
    expression = get_prediction_label(img)
    show_alert(expression)

st.header("📼 Upload Video")
uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    st.info("⏳ Processing video...")
    last_expression = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        expression = get_prediction_label(frame)
        last_expression = expression
        cv2.putText(frame, f"{expression.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        stframe.image(frame, channels="BGR")

    cap.release()
    if last_expression:
        show_alert(last_expression)
