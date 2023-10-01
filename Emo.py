import numpy as np
import streamlit as st
import pandas as pd
import cv2
import random
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D

# Load the Muse v3 data from the CSV file
df = pd.read_csv('muse_v3.csv')

def filter_tracks(emotion, genre):
    filtered_tracks = df[df['genre'] == genre]
    return filtered_tracks

def preprocess_emotions(emotions):
    result = [item for items, c in Counter(emotions).most_common() for item in [items] * c]
    unique_emotions = list(set(result))
    return unique_emotions

# Load the emotion detection model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.load_weights('model.h5')

# Define emotion mapping dictionary
emotion_dict = {0: "angry", 1: "disgusted", 2: "fearful", 3: "happy", 4: "neutral", 5: "sad", 6: "surprised"}

# Define Streamlit app title style
st.markdown(
    """
    <style>
        .title {
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 20px;
            color: #0072b1; /* Title color */
        }
        .twinkling-stars {
            background: linear-gradient(rgba(0, 0, 0, 0.2) 1px, transparent 1px),
                        linear-gradient(90deg, rgba(0, 0, 0, 0.2) 1px, transparent 1px);
            background-size: 20px 20px;
            width: 100%;
            height: 100vh;
            animation: twinkling 10s linear infinite;
        }
         body {
            background-color: #FFABB8; /* Change the background color here */
        }
        @keyframes twinkling {
            0% { background-position: 0 0; }
            100% { background-position: 20px 20px; }
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='title'>Emotion-Based Music Recommender</h1>", unsafe_allow_html=True)

# Create Streamlit app layout columns
col1, col2, col3 = st.columns(3)

# Create the video capture object 'cap' before entering the loop
cap = cv2.VideoCapture(0)

# Define an empty list called list_emotions
list_emotions = []

with col2:
    if st.button('SCAN EMOTION (Click here)'):
        count = 0
        list_emotions.clear()
        max_frames = 20
        capturing = True

        while capturing:
            ret, frame = cap.read()
            if not ret:
                break

            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            count += 1

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 100), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]

                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

                prediction = model.predict(cropped_img)

                max_index = int(np.argmax(prediction))

                if max_index in emotion_dict:
                    list_emotions.append(emotion_dict[max_index])
                else:
                    print(f"Invalid emotion prediction: {max_index}")

                cv2.putText(frame, emotion_dict[max_index], (x + 20, y - 60), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)
                cv2.imshow('Video', cv2.resize(frame, (1000, 700), interpolation=cv2.INTER_CUBIC))

                if cv2.waitKey(1) & 0xFF == ord('x'):
                    capturing = False  # Stop capturing if 'x' is pressed
                    break

            if count >= max_frames:
                capturing = False  # Stop capturing after reaching max frames

        cap.release()
        cv2.destroyAllWindows()

# Define a dictionary to map emotions to genres
# Define emotion-genre mapping with multiple genres for each emotion
emotion_genre_mapping = {
    'happy': ['rock', 'rap', 'pop'],
    'sad': ['pop'],
    'angry': ['punk', ],
    'neutral': ['rock', 'pop', 'hip-hop'],
    'surprised': ['rap', 'hip-hop']
}

# Manually enter emotion
manual_emotion = st.text_input("Enter your emotion manually (e.g., happy, sad, angry):")

# Style the input box
style = """
<style>
    .manual-emotion-input {
        padding: 10px;
        border: 2px solid #0072b1; /* Border color */
        border-radius: 5px;
        font-size: 30px;
        color: #0072b1; /* Text color */
        background-color: transparent;
    }
</style>
"""
st.markdown(style, unsafe_allow_html=True)

if manual_emotion:
    manual_emotion = manual_emotion.lower()

    if manual_emotion in emotion_genre_mapping:
        # ... Rest of your code

        genres_to_filter = emotion_genre_mapping[manual_emotion]

        # You can suggest multiple genres
        for genre in genres_to_filter:
            filtered_tracks = filter_tracks(manual_emotion, genre)

            st.write(f"Suggested {genre.capitalize()} Songs for Emotion:", manual_emotion)

            style = """
            <style>
                .suggested-songs {
                    background-color: white;
                    padding: 10px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }
                .song-info {
                    font-size: 18px;
                    margin-bottom: 10px;
                }
                .song-info strong {
                    color: black;
                    font-weight: bold;
                }
                .song-info a {
                    text-decoration: none;
                    color: #0072b1; /* Link color */
                }
                .artist-name {
                    color: #0072b1; /* Artist name color */
                }
                .genre-name {
                    color: #0072b1; /* Genre color */
                }
            </style>
            """
            st.markdown(style, unsafe_allow_html=True)

            for index, row in filtered_tracks.iterrows():
                track_info = f"<div class='suggested-songs'><p class='song-info'><strong>Track:</strong> <a href='{row['lastfm_url']}'>{row['track']}</a></p><p class='song-info'><strong>Artist:</strong> <span class='artist-name'>{row['artist']}</span></p><p class='song-info'><strong>Genre:</strong> <span class='genre-name'>{row['genre']}</span></p></div>"
                st.markdown(track_info, unsafe_allow_html=True)
    else:
        st.write("Genre not found for manually entered emotion:", manual_emotion)
elif list_emotions:
    captured_emotion = Counter(list_emotions).most_common(1)[0][0]

    if captured_emotion in emotion_genre_mapping:
        genres_to_filter = emotion_genre_mapping[captured_emotion]

        # You can suggest multiple genres
        for genre in genres_to_filter:
            filtered_tracks = filter_tracks(captured_emotion, genre)

            st.write(f"Suggested {genre.capitalize()} Songs for Emotion:", captured_emotion)

            style = """
            <style>
                .suggested-songs {
                    background-color: white;
                    padding: 10px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }
                .song-info {
                    font-size: 18px;
                    margin-bottom: 10px;
                }
                .song-info strong {
                    color: black;
                    font-weight: bold;
                }
                .song-info a {
                    text-decoration: none;
                    color: #0072b1; /* Link color */
                }
                .artist-name {
                    color: #0072b1; /* Artist name color */
                }
                .genre-name {
                    color: #0072b1; /* Genre color */
                }
            </style>
            """
            st.markdown(style, unsafe_allow_html=True)

            for index, row in filtered_tracks.iterrows():
                track_info = f"<div class='suggested-songs'><p class='song-info'><strong>Track:</strong> <a href='{row['lastfm_url']}'>{row['track']}</a></p><p class='song-info'><strong>Artist:</strong> <span class='artist-name'>{row['artist']}</span></p><p class='song-info'><strong>Genre:</strong> <span class='genre-name'>{row['genre']}</span></p></div>"
                st.markdown(track_info, unsafe_allow_html=True)
    else:
        st.write("Genre not found for captured emotion:", captured_emotion)
else:
    st.write("No emotion captured.")
