import os
import streamlit as st
import whisper
import ffmpeg
import numpy as np
from moviepy.editor import VideoFileClip
import uuid
import speech_recognition as sr
from pytube import YouTube
import openai 
from streamlit import session_state as st_session_state

# Set your OpenAI API key
openai.api_key = "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"

# Set app wide config
st.set_page_config(
    page_title="Visa Skip",
    page_icon="🤖",
    menu_items={
        "About": """This project implements an interface for searching videos using Open AI [Whisper] Models.""",
    },
)

# Function to load Whisper model
@st.cache_data
def load_model():
    return whisper.load_model("tiny.en.pt")

# Function to load audio file
@st.cache_data
def load_audio(file: (str, bytes), sr: int = 16000):
    if isinstance(file, bytes):
        inp = file
        file = 'pipe:'
    else:
        inp = None

    try:
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True, input=inp)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

# Function to search for query in transcribed audio
@st.cache_data
def searcher(trans_dict, query):
    results = []
    segments = trans_dict['segments']

    for segment in segments:
        if query.lower() in segment['text'].lower():
            start_m, start_s = divmod(int(segment['start']), 60)
            end_m, end_s = divmod(int(segment['end']), 60)
            results.append(f'{start_m:02d}:{start_s:02d} - {end_m:02d}:{end_s:02d}')

    return results

# Function to transcribe audio
@st.cache_data
def transcribe(_model, audio_array):
    return _model.transcribe(audio_array, language='english')

# Function to display columns of search results
def col_displayer(lst, wcol=6):
    ncol = len(lst)
    cols = st.columns(ncol)

    for i in range(ncol):
        col = cols[i % wcol]
        col.write(f"{lst[i]}")

    return cols

# Function to get YouTube video bytes
@st.cache_data
def get_yt_bytes(yt_link):
    audio = YouTube(yt_link).streams.filter(only_audio=True).first().download()
    with open(audio, "rb") as f:
        _bytes = f.read()
    os.remove(audio)
    return _bytes

# Streamlit UI
def main():
    # Initialize session state
    if "found_text" not in st_session_state:
        st_session_state.found_text = None

    # Add the left sidebar
    st.sidebar.subheader('About')
    st.sidebar.write("""This project implements an interface for searching videos using Open AI [Whisper] Models.""")

    # Set the background color for the sidebar
    st.sidebar.markdown(
        """
        <style>
            .sidebar-content {
                background-color: #f8f9fa;
                padding: 1.5rem;
                border-radius: 5px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<span style="color: blue; font-size: 100px;">Visa</span> <span style="color: yellow; font-size:100px;">Skip</span>', unsafe_allow_html=True)

    'Made by Ahaz Bhatti'

    # Add logo if available

    upload_type = st.radio(
        "What kind of video do you wanna search?",
        ('YouTube Vid', 'Local Video')
    )

    if upload_type == 'YouTube Vid':
        yt_link = st.text_input("Enter a YouTube Video Link")
        if len(yt_link) > 10:
            audio_bytes = get_yt_bytes(yt_link)

            # Text Input
            query = st.text_input("Enter some query 👇")

            # audio_bytes are the bytes of the audio file
            audio_array = load_audio(audio_bytes)

            clicked = st.button('Search')

            # Search button
            if clicked:
                if len(query) > 1:
                    data_load_state = st.text('Searching...')

                    trans_dict = transcribe(load_model(), audio_array)

                    search_result = searcher(trans_dict, query)

                    data_load_state.text('Search.. done!')

                    if search_result:
                        st.success(f"We found '{query}' at the following position(s):")
                        col_displayer(search_result)

                        # Extract found text
                        st_session_state.found_text = " ".join(segment['text'] for segment in trans_dict['segments'])

                    else:
                        st.warning(f"We couldn't find '{query}'")

    else:
        # Upload
        uploaded_file = st.file_uploader("Upload a file", type=["mp3", "ogg", "wav", "aac", "m4a", "flac", "avi", "wma", "mp4", "mkv", "mov", "wmv"])
        if uploaded_file is not None:
            # Display uploaded video
            st.video(uploaded_file)

            # To read file as bytes:
            audio_bytes = uploaded_file.getvalue()

            # Text Input
            query = st.text_input("Enter some query 👇")

            # audio_bytes are the bytes of the audio file
            audio_array = load_audio(audio_bytes)

            clicked = st.button('Search')

            # Search button
            if clicked:
                if len(query) > 1:
                    data_load_state = st.text('Searching...')

                    trans_dict = transcribe(load_model(), audio_array)

                    search_result = searcher(trans_dict, query)

                    data_load_state.text('Search.. done!')

                    if search_result:
                        st.success(f"We found '{query}' at the following position(s):")
                        col_displayer(search_result)

                        # Extract found text
                        st_session_state.found_text = " ".join(segment['text'] for segment in trans_dict['segments'])

                    else:
                        st.warning(f"We couldn't find '{query}'")

    # Display found text with a toggle button
    with st.expander("Found Text in the Video", expanded=True):
        if st_session_state.found_text:
            st.write(st_session_state.found_text)

        # CO-Pilot Analysis button
        if st.button("CO-Pilot Analysis"):
            if st_session_state.found_text:
                st.write("Performing CO-Pilot Analysis...")

                # Add your CO-Pilot analysis logic here
                prompt = f"Summarize the following text:\n\n{st_session_state.found_text}"
                response = openai.completions.create(
                    model="text-davinci-003",  # You can use other engines too
                    prompt=prompt,
                    max_tokens=150,
                    temperature=0.5,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )

                generated_summary = response.choices[0].text.strip()

                st.write("Generated Summary:")
                st.write(generated_summary)
            else:
                st.warning("No found text available for CO-Pilot Analysis.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
