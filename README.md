
# EDA_App

A no-code Streamlit app for fast, interactive exploratory data analysis and transformation of tabular datasets using ydata-profiling.

## Features

- **File Upload:** Upload `.csv`, `.xls`, or `.xlsx` files. Instantly preview your data, see its shape, missing values, and inferred data types.
- **Data Type Handling:** View and manually change column data types (e.g., convert object to datetime).
- **Column Filtering:** Select and exclude columns from the EDA report.
- **EDA Report with ydata-profiling:** Generate a full or minimal HTML profiling report. Open in a new tab or download as a standalone HTML file.
- **Data Transformations:**
  - Log transformation
  - Standardization (z-score)
  - Normalization (Min-Max)
  - Missing value handling (drop, fill mean/median/mode)
  - Outlier removal (Z-Score, IQR)
  - Categorical encoding (label or one-hot, including numeric columns)
- **Transformed Dataset Output:** Download the updated dataset (CSV/Excel) and generate a new profiling report for the transformed data.
- **User Interface:** Sidebar navigation, dark mode toggle, and a Help tab for guidance.

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sablania-dev/EDA_App.git
   cd EDA_App/EDA_App
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Or manually install:
   ```bash
   pip install streamlit ydata-profiling pandas numpy scikit-learn openpyxl
   ```
3. **Run the app:**
   ```bash
   streamlit run main.py
   ```

## Usage

1. **Upload Data:** Upload your dataset and review its structure and types.
2. **Transform Data:** Apply transformations, handle missing values, remove outliers, and encode categorical/numeric columns.
3. **EDA Report:** Generate and download a profiling report for your selected columns.
4. **Download Data:** Download the transformed dataset and its profiling report.
5. **Help:** Find usage instructions and troubleshooting tips.
6. **About the Creator:** Learn more about the author and connect.

## Author

**Sarthak Sablania**  
Data Scientist | IIT Delhi BTech 2020 | IIM Calcutta Masters 2026

- [LinkedIn](https://www.linkedin.com/in/sarthak-sablania/)
- [GitHub](https://github.com/sablania-dev/)

Feel free to reach out for feedback, suggestions, or collaboration!
