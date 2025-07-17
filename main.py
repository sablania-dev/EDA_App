
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
page = st.sidebar.radio("Go to", ["Upload Data", "Transform Data", "EDA Report", "Download Data", "Help", "About the Creator"])
def about_tab():
    st.title("About the Creator")
    st.markdown("""
**Sarthak Sablania**

*Data Scientist | IIT Delhi BTech 2020 | IIM Calcutta Masters 2026*

Hi! I'm Sarthak Sablania. As a data scientist, I often found it tedious to write code for every small EDA task. That's why I created this app—to make exploratory data analysis easier, faster, and more accessible for everyone.

**Connect with me:**
- [LinkedIn](https://www.linkedin.com/in/sarthak-sablania/)
- [GitHub](https://github.com/sablania-dev/)

Feel free to reach out for feedback, suggestions, or collaboration!
    """)

# --- Dark Mode Toggle ---
dark_mode = st.sidebar.checkbox("Dark Mode", value=False)
if dark_mode:
    st.markdown("""
        <style>
        body, .stApp { background-color: #222; color: #eee; }
        .css-1d391kg { background-color: #222; }
        </style>
    """, unsafe_allow_html=True)

# --- Session State ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'transformed_df' not in st.session_state:
    st.session_state.transformed_df = None
if 'profile_html' not in st.session_state:
    st.session_state.profile_html = None
if 'profile_html_trans' not in st.session_state:
    st.session_state.profile_html_trans = None

def file_uploader():
    st.title("Upload your dataset")
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xls", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.session_state.df = df.copy()
        st.session_state.transformed_df = df.copy()
        # Store original filename (without extension)
        st.session_state.original_filename = uploaded_file.name.rsplit('.', 1)[0]
        st.write("### Data Preview", df.head())
        st.write(f"**Shape:** {df.shape}")
        st.write("**Missing Values:**")
        st.write(df.isnull().sum())
        st.write("**Inferred Data Types:**")
        st.write(df.dtypes)
        # Data type editing
        st.write("#### Change Data Types")
        dtype_map = {col: str(df[col].dtype) for col in df.columns}
        new_types = {}
        for col, dtype in dtype_map.items():
            options = [dtype, "int64", "float64", "object", "datetime64[ns]"]
            new_type = st.selectbox(f"{col}", options, index=options.index(dtype) if dtype in options else 0, key=f"dtype_{col}")
            new_types[col] = new_type
        if st.button("Apply Data Types"):
            for col, typ in new_types.items():
                try:
                    if typ == "datetime64[ns]":
                        st.session_state.df[col] = pd.to_datetime(st.session_state.df[col], errors='coerce')
                    else:
                        st.session_state.df[col] = st.session_state.df[col].astype(typ)
                except Exception as e:
                    st.warning(f"Could not convert {col}: {e}")
            st.session_state.transformed_df = st.session_state.df.copy()
            st.rerun()

def column_filter():
    st.write("### Column Filtering")
    df = st.session_state.df
    columns = list(df.columns)
    selected = st.multiselect("Select columns to include in EDA report", columns, default=columns)
    if st.button("Apply Column Filter"):
        st.session_state.filtered_columns = selected
        st.success("Column selection updated.")
        st.experimental_rerun()
    return st.session_state.get('filtered_columns', columns)

def generate_profile(df, minimal=False):
    profile = ProfileReport(df, minimal=minimal, explorative=True)
    return profile.to_html()

def eda_report():
    st.title("EDA Report")
    df = st.session_state.df
    if df is None:
        st.warning("Please upload data first.")
        return
    columns = column_filter()
    df = df[columns]
    st.session_state.df = df
    st.write("#### Choose Report Type")
    minimal = st.radio("Report Type", ["Full", "Minimal"], horizontal=True) == "Minimal"
    if st.button("Generate Report") or st.session_state.profile_html is None:
        with st.spinner("Generating report..."):
            html = generate_profile(df, minimal)
            st.session_state.profile_html = html
    if st.session_state.profile_html:
        st.markdown("[Open Report in New Tab](data:text/html;base64,{})".format(base64.b64encode(st.session_state.profile_html.encode()).decode()), unsafe_allow_html=True)
        st.download_button("Download Report as HTML", st.session_state.profile_html, file_name="EDA_Report.html")

def log_transform(df, cols):
    for col in cols:
        df[col] = np.log1p(df[col])
    return df

def standardize(df, cols):
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df

def normalize(df, cols):
    scaler = MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df

def handle_missing(df, method, cols=None):
    if method == "Drop Rows":
        return df.dropna()
    elif method == "Fill Mean":
        return df.fillna(df.mean(numeric_only=True))
    elif method == "Fill Median":
        return df.fillna(df.median(numeric_only=True))
    elif method == "Fill Mode":
        return df.fillna(df.mode().iloc[0])
    return df

def remove_outliers(df, method, cols):
    if method == "Z-Score":
        from scipy.stats import zscore
        z = np.abs(zscore(df[cols], nan_policy='omit'))
        return df[(z < 3).all(axis=1)]
    elif method == "IQR":
        Q1 = df[cols].quantile(0.25)
        Q3 = df[cols].quantile(0.75)
        IQR = Q3 - Q1
        mask = ~((df[cols] < (Q1 - 1.5 * IQR)) | (df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)
        return df[mask]
    return df

def encode_categorical(df, cols, method):
    if method == "Label Encoding":
        le = LabelEncoder()
        for col in cols:
            df[col] = le.fit_transform(df[col].astype(str))
    elif method == "One-Hot Encoding":
        df = pd.get_dummies(df, columns=cols)
    return df

def transform_data():
    st.title("Data Transformations")
    df = st.session_state.transformed_df.copy()
    st.write("#### Data Preview", df.head())
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    all_encodable_cols = cat_cols + [col for col in num_cols if col not in cat_cols]

    # Log Transform
    log_cols = st.multiselect("Select columns for Log Transform", num_cols, key="log_cols")
    if st.button("Apply Log Transform") and log_cols:
        df = log_transform(df, log_cols)
        st.session_state.transformed_df = df.copy()
        st.success("Log transform applied.")
        st.rerun()

    # Standardization
    std_cols = st.multiselect("Select columns for Standardization (z-score)", num_cols, key="std_cols")
    if st.button("Apply Standardization") and std_cols:
        df = standardize(df, std_cols)
        st.session_state.transformed_df = df.copy()
        st.success("Standardization applied.")
        st.rerun()

    # Normalization
    norm_cols = st.multiselect("Select columns for Normalization (Min-Max)", num_cols, key="norm_cols")
    if st.button("Apply Normalization") and norm_cols:
        df = normalize(df, norm_cols)
        st.session_state.transformed_df = df.copy()
        st.success("Normalization applied.")
        st.rerun()

    # Missing Value Handling
    st.write("#### Handle Missing Values")
    missing_method = st.selectbox("Method", ["None", "Drop Rows", "Fill Mean", "Fill Median", "Fill Mode"], key="missing_method")
    if st.button("Apply Missing Value Handling") and missing_method != "None":
        df = handle_missing(df, missing_method)
        st.session_state.transformed_df = df.copy()
        st.success(f"Missing value handling ({missing_method}) applied.")
        st.rerun()

    # Outlier Removal
    st.write("#### Remove Outliers")
    outlier_method = st.selectbox("Outlier Method", ["None", "Z-Score", "IQR"], key="outlier_method")
    outlier_cols = st.multiselect("Columns for Outlier Removal", num_cols, key="outlier_cols")
    if st.button("Apply Outlier Removal") and outlier_method != "None" and outlier_cols:
        df = remove_outliers(df, outlier_method, outlier_cols)
        st.session_state.transformed_df = df.copy()
        st.success(f"Outlier removal ({outlier_method}) applied.")
        st.rerun()

    # Categorical Encoding
    st.write("#### Encode Categorical Variables")
    enc_method = st.selectbox("Encoding Method", ["None", "Label Encoding", "One-Hot Encoding"], key="enc_method")
    enc_cols = st.multiselect("Columns to Encode (categorical or numeric)", all_encodable_cols, key="enc_cols")
    if st.button("Apply Encoding") and enc_method != "None" and enc_cols:
        df = encode_categorical(df, enc_cols, enc_method)
        st.session_state.transformed_df = df.copy()
        st.success(f"Categorical encoding ({enc_method}) applied.")
        st.rerun()

    if st.button("Save Transformed Data"):
        st.session_state.transformed_df = df.copy()
        st.success("Transformed data saved.")
        st.rerun()

def download_data():
    st.title("Download Data")
    df = st.session_state.transformed_df
    if df is None:
        st.warning("No transformed data available.")
        return
    st.write("#### Preview", df.head())
    # Determine transformed filename
    base_name = st.session_state.get('original_filename', 'transformed_data')
    csv_name = f"{base_name}_transformed.csv"
    xlsx_name = f"{base_name}_transformed.xlsx"
    csv = df.to_csv(index=False).encode()
    st.download_button("Download CSV", csv, file_name=csv_name)
    excel_buffer = BytesIO()
    df.to_excel(excel_buffer, index=False, engine='openpyxl')
    st.download_button("Download Excel", excel_buffer.getvalue(), file_name=xlsx_name)
    # Generate new profiling report for transformed data
    if st.button("Generate EDA Report for Transformed Data") or st.session_state.profile_html_trans is None:
        with st.spinner("Generating report..."):
            html = generate_profile(df)
            st.session_state.profile_html_trans = html
    if st.session_state.profile_html_trans:
        st.markdown("[Open Transformed Report in New Tab](data:text/html;base64,{})".format(base64.b64encode(st.session_state.profile_html_trans.encode()).decode()), unsafe_allow_html=True)
        st.download_button("Download Transformed Report as HTML", st.session_state.profile_html_trans, file_name="Transformed_EDA_Report.html")


def help_tab():
    st.title("Help & Usage Guide")
    st.markdown("""
**Welcome to the EDA App!**

**How to use:**

1. **Upload Data**
   - Upload your dataset in `.csv`, `.xls`, or `.xlsx` format.
   - Preview your data, check its shape, missing values, and inferred data types.
   - Change column data types if needed.

2. **Transform Data**
   - Apply log transformation, standardization, or normalization to numeric columns.
   - Handle missing values by dropping rows or filling with mean, median, or mode.
   - Remove outliers using Z-Score or IQR methods.
   - Encode categorical columns using label or one-hot encoding.
   - Save your transformed data for further analysis.

3. **EDA Report**
   - Select columns to include in the report.
   - Choose between a full or minimal profiling report.
   - Generate, view, and download the EDA report as an HTML file.

4. **Download Data**
   - Download your transformed dataset as CSV or Excel.
   - Generate and download a profiling report for the transformed data.

**Tips:**
- Use the sidebar to navigate between steps.
- Toggle dark mode for a comfortable viewing experience.
- All changes are in-memory; re-upload your data to reset.
- For large files, processing may take a few moments.
- If you encounter errors, check your data types and missing values.

**Need more help?**
- Visit the [ydata-profiling documentation](https://ydata-profiling.ydata.ai/docs/master/) for details on EDA reports.
- For Streamlit usage, see the [Streamlit docs](https://docs.streamlit.io/).
    """)

# --- Main App Logic ---
if page == "Upload Data":
    file_uploader()
    st.sidebar.button("Go to Transform Data", on_click=lambda: st.session_state.update({'page': 'Transform Data'}))
elif page == "Transform Data":
    if st.session_state.df is not None:
        transform_data()
    else:
        st.warning("Please upload data first.")
elif page == "EDA Report":
    if st.session_state.df is not None:
        eda_report()
    else:
        st.warning("Please upload data first.")
elif page == "Download Data":
    if st.session_state.transformed_df is not None:
        download_data()
    else:
        st.warning("Please upload and transform data first.")
elif page == "Help":
    help_tab()
elif page == "About the Creator":
    about_tab()
