import streamlit as st
import os

st.title("🔍 System Diagnostic")

# Check if the files actually exist in the cloud environment
st.write("Checking for files...")
files = os.listdir('.')
st.write(f"Files found in directory: {files}")

if 'boundaries.geojson' in files:
    st.success("✅ boundaries.geojson found!")
else:
    st.error("❌ boundaries.geojson NOT found!")

if 'requirements.txt' in files:
    st.success("✅ requirements.txt found!")
else:
    st.error("❌ requirements.txt NOT found!")

st.write("If you can see this, the app is running! Now check the logs on the right to see if the libraries installed correctly.")