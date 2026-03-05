import streamlit as st
import time
import pandas as pd
import tempfile
import os
import whisper

# 1. Page Configuration
st.set_page_config(page_title="ClassifAI", page_icon="🎓", layout="centered")

# 2. Header Section
st.title("ClassifAI 🎓")
st.subheader("AI-Driven Instructional Feedback System")
st.write("Upload a classroom recording to analyze talking distribution and question categories.")

# 3. File Uploader
uploaded_file = st.file_uploader("Drag and drop classroom audio file here", type=["wav", "mp3", "m4a"])

# 4. Action Button
if st.button("Analyze Audio", type="primary", use_container_width=True):
    if uploaded_file is not None:

        # Simulate the time it takes to process audio
        with st.spinner("Processing audio... (Diarization & Transcription)"):
            time.sleep(2) # Pauses for 2 seconds to simulate work

        # Save the uploaded file to a temporary file on the disk
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name # Save the path to hand to Whisper
            
        st.success("File saved temporarily! Ready for Whisper.")
        
        # We will put the Whisper code here next...
        with st.spinner("Transcribing audio with Whisper... this might take a minute."):
            # Load the 'base' model (good balance of speed and accuracy for a prototype)
            model = whisper.load_model("base")
            
            # Transcribe the audio
            result = model.transcribe(tmp_file_path)
            
            st.write("### Raw Transcript with Timestamps")
            
            # Loop through the segments to show the timestamps
            for segment in result["segments"]:
                start = segment['start']
                end = segment['end']
                text = segment['text']
                
                # Display it nicely in Streamlit
                st.write(f"**[{start:.2f}s - {end:.2f}s]:** {text}")
            
        st.success("Analysis Complete!")
        
        st.divider()
        
        # 5. Results Section: Talking Distribution
        st.header("Analysis Results")
        st.subheader("Talking Distribution")
        
        # Using Streamlit columns to display stats side-by-side
        col1, col2 = st.columns(2)
        col1.metric(label="Teacher Talk Time", value="65%")
        col2.metric(label="Student Talk Time", value="35%")
        
        # Visual progress bar for the ratio
        st.progress(65, text="Teacher vs. Student Ratio")
        
        st.divider()
        
        # 6. Results Section: Question Categorization
        st.subheader("Question Categorization (Costa's Levels)")
        
        # Creating a placeholder dataframe to mimic the final output
        mock_data = {
            "Timestamp": ["04:12", "08:45", "14:20"],
            "Question Asked": [
                "What is the output of this specific line of code?",
                "How does this approach compare to the one we used last week?",
                "If we change the algorithm, predict what will happen to the efficiency."
            ],
            "Costa's Level": [
                "Level 1 (Gathering)", 
                "Level 2 (Processing)", 
                "Level 3 (Applying)"
            ]
        }
        
        df = pd.DataFrame(mock_data)
        
        # Display the table cleanly
        st.dataframe(df, use_container_width=True, hide_index=True)


        # Cleanup: Delete the file when we are done
        os.remove(tmp_file_path)
        
    else:
        st.warning("Please upload an audio file first!")