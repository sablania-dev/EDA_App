
import streamlit as st
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
from io import BytesIO
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import base64

st.set_page_config(page_title="EDA App", layout="wide", initial_sidebar_state="expanded")

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["EDA", "Help", "About"])
def about_tab():
    st.title("About the Creator")
    st.markdown("""
**Sarthak Sablania**

*Data Scientist | IIT Delhi BTech 2020 | IIM Calcutta Masters 2026*

Hi! I'm Sarthak Sablania. I created this app to make EDA effortless and accessible for everyone—no coding required!

**Connect with me:**
- [LinkedIn](https://www.linkedin.com/in/sarthak-sablania/)
- [GitHub](https://github.com/sablania-dev/)

Feel free to reach out for feedback, suggestions, or collaboration!
    """)


# --- Session State ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'transformed_df' not in st.session_state:
    st.session_state.transformed_df = None
if 'profile_html' not in st.session_state:
    st.session_state.profile_html = None
if 'profile_html_trans' not in st.session_state:
    st.session_state.profile_html_trans = None


def eda_tab():
    st.title("Quick EDA Report")
    uploaded_file = st.file_uploader("Upload a CSV, XLS, or XLSX file", type=["csv", "xls", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.write("### Data Preview", df.head())
        st.write(f"**Shape:** {df.shape}")
        if st.button("Generate EDA Report"):
            progress = st.progress(0, text="Starting report generation...")
            import time
            for i in range(1, 6):
                time.sleep(0.2)
                progress.progress(i * 20, text=f"Generating report... ({i*20}%)")
            with st.spinner("Generating report..."):
                profile = ProfileReport(df, explorative=True)
                html = profile.to_html()
                st.session_state.profile_html = html
            progress.progress(100, text="Report generation complete!")
            time.sleep(0.5)
            progress.empty()
        if st.session_state.get("profile_html"):
            st.markdown("[Open EDA Report in New Tab](data:text/html;base64,{})".format(base64.b64encode(st.session_state.profile_html.encode()).decode()), unsafe_allow_html=True)
            st.download_button("Download EDA Report as HTML", st.session_state.profile_html, file_name="EDA_Report.html")


def help_tab():
    st.title("Help & Usage Guide")
    st.markdown("""
**How to use this app:**

1. **Go to the EDA tab.**
2. **Upload your data file** (`.csv`, `.xls`, or `.xlsx`).
3. **Click 'Generate EDA Report'** to create a ydata-profiling report.
4. **Open or download the report** using the provided link or button.

**Tips:**
- Use the sidebar to switch between EDA, Help, and About tabs.
- For large files, report generation may take some time.

**More info:**
- [ydata-profiling documentation](https://ydata-profiling.ydata.ai/docs/master/)
- [Streamlit documentation](https://docs.streamlit.io/)
    """)

# --- Main App Logic ---
if page == "EDA":
    eda_tab()
elif page == "Help":
    help_tab()
elif page == "About":
    about_tab()
