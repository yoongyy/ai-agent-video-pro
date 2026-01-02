# AI Property Video Generator

This is a Streamlit application that uses AI to generate a video tour of a property.

## Setup

1.  Install the dependencies:
    ```
    pip install -r requirements.txt
    ```
2.  Create a `.env` file and add your API keys:
    ```
    gemini_key=YOUR_GEMINI_API_KEY
    elevenLabs_key=YOUR_ELEVENLABS_API_KEY
    VOICE_ID=YOUR_VOICE_ID
    ```
3.  Run the application:
    ```
    streamlit run app.py
    ```

## Usage

1.  Open the application in your browser.
2.  Paste the property information into the text area.
3.  Upload 3-5 images of the property.
4.  Click the "Generate AI Video" button.
5.  The application will generate a video tour of the property with a voiceover.
6.  You can then download the video.
