import streamlit as st 
import pandas as pd
import tempfile
import re
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import os

spark = SparkSession.builder.appName("StreamlitPySparkApp").getOrCreate()
st.header('Dataset Cleansing App', divider='rainbow')
st.markdown("# Upload")

def sanitize_column_names(df):
    def to_snake_case(name):
        # Replace camelCase or PascalCase with snake_case
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name)
        # Replace all non-word characters and leading digits
        name = re.sub(r'\W|^(?=\d)', '_', name)
        # Convert to lower case
        return name.lower()
    
    sanitized_columns = [to_snake_case(col) for col in df.columns]
    for old_col, new_col in zip(df.columns, sanitized_columns):
        df = df.withColumnRenamed(old_col, new_col)
    return df

def sanitize_columns(df):
    return sanitize_column_names(df)

def read_file(file):
    if file is not None:
        file_type = file.name.split('.')[-1]
        
        # Save file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}') as temp_file:
            temp_file.write(file.getvalue())
            temp_file_path = temp_file.name

        if file_type == 'csv':
            return spark.read.csv(temp_file_path, header=True, inferSchema=True)
        elif file_type == 'json':
            return spark.read.json(temp_file_path)
        elif file_type in ['xls', 'xlsx']:
            pd_df = pd.read_excel(temp_file_path)
            return spark.createDataFrame(pd_df)
        else:
            st.error("Unsupported file type!")
        
        # Clean up the temporary file
        os.remove(temp_file_path)
    return None

def dataset_summary(df):
    # 1. Categorical, Numerical, Ordinal, and Time Variables Identification
    dtypes = df.dtypes
    categorical = [name for name, dtype in dtypes if dtype == 'string']
    numerical = [name for name, dtype in dtypes if dtype in ['int', 'bigint', 'double', 'float']]
    time = [name for name, dtype in dtypes if dtype == 'timestamp']
    
    st.write("Categorical Variables:", categorical)
    st.write("Numerical Variables:", numerical)
    st.write("Time Variables:", time)

    # 2. Data Types of Each Column
    st.write("Data Types:")
    st.write(pd.DataFrame(dtypes, columns=['Column', 'Data Type']))

    # 3. Columns with Null Values and Their Counts
    null_counts_list = []
    for column in df.columns:
        null_count = df.filter(F.col(f"`{column}`").isNull()).count()
        null_counts_list.append((column, null_count))
    
    null_counts_df = pd.DataFrame(null_counts_list, columns=['Column', 'Null Count'])
    st.write("Columns with Null Values:")
    st.write(null_counts_df)

    # 4. Outliers for Numerical Variables
    outliers = {}
    for col in numerical:
        q1, q3 = df.approxQuantile(f"`{col}`", [0.25, 0.75], 0.05)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = df.filter((F.col(f"`{col}`") < lower_bound) | (F.col(f"`{col}`") > upper_bound)).count()
        outliers[col] = outlier_count
    
    st.write("Outliers for Numerical Variables:")
    st.write(pd.DataFrame(list(outliers.items()), columns=['Column', 'Outlier Count']))

    # st.write("Summary Statistics:")
    # summary_df = df.summary()
    # st.write(summary_df)

def clean_data(df, actions):
    for action in actions:
        col = action['column']
        
        # Handle null values
        if action['null_action'] == "Drop rows with nulls":
            df = df.filter(F.col(col).isNotNull())
        elif action['null_action'] == "Fill nulls with value":
            if action['fill_value']:
                df = df.fillna({col: action['fill_value']})
        elif action['null_action'] == "Fill nulls with average":
            avg_value = df.select(F.mean(F.col(col))).first()[0]
            df = df.fillna({col: avg_value})
        
        # # Handle outliers for numerical columns
        # if action['outlier_action'] == "Remove outliers":
        #     q1, q3 = df.approxQuantile(col, [0.25, 0.75], 0.05)
        #     iqr = q3 - q1
        #     lower_bound = q1 - 1.5 * iqr
        #     upper_bound = q3 + 1.5 * iqr
        #     df = df.filter((F.col(col) >= lower_bound) & (F.col(col) <= upper_bound))

    return df

def get_cleaning_actions(df):
    st.markdown("# Data Cleaning Options")
    
    # Get the columns from the dataframe
    columns = df.columns
    
    # Store the cleaning actions
    actions = []

    # Display cleaning options for each column
    for col in columns:
        st.markdown(f"### Cleaning Options for `{col}`")
        
        # Options for handling null values
        null_action = st.selectbox(f"Handle null values in `{col}`:", ["None", "Drop rows with nulls", "Fill nulls with value", "Fill nulls with average"], key=f"null_{col}")
        fill_value = None
        if null_action == "Fill nulls with value":
            fill_value = st.text_input(f"Value to fill nulls in `{col}`:", key=f"fill_{col}")
        
        # Collect actions
        actions.append({
            'column': col,
            'null_action': null_action,
            'fill_value': fill_value
        })
    
    return actions

# def export_data(df, filename):
#     # Use a temporary file to save the CSV
#     # with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
#     #     temp_path = temp_file.name
#     #     # Save DataFrame to CSV
#     #     df.toPandas().to_csv(temp_path, index=False)
    
    
#     # Create download link for the CSV file
#     # st.markdown(f"[Download {filename}]({temp_path})")


file = st.file_uploader("Upload your file", type=['csv', 'json', 'xls', 'xlsx'])
df = read_file(file)

if df:
    st.write('\n')
    st.write('Data Overview:')
    st.write('\n')
    st.dataframe(df.toPandas())
    st.write('\n')
    st.write('Shape: ',(df.count(), len(df.columns)),'')

st.markdown("# Data Summary")

if df:
    dataset_summary(df)

if df:
    df = sanitize_columns(df)
    st.write('\n')
    st.write('Columns renamed:')
    st.write('\n')
    st.dataframe(df.toPandas())

st.markdown("# Data Cleaning")

# Get cleaning actions but do not apply them immediately
if df:
    actions = get_cleaning_actions(df)

# Button to apply cleaning
if df:
    if st.button("Clean"):
        st.markdown("# Cleaned Data")
        df = clean_data(df, actions)
        st.dataframe(df.toPandas())
        st.write('\n')
        st.write('New Shape: ',(df.count(), len(df.columns)),'')

        # Get the output filename from the user
        filename = st.text_input("Enter the filename for the cleaned data:", "cleaned_data.csv")

        # Convert DataFrame to CSV for download
        csv_data = df.toPandas().to_csv(index=False).encode('utf-8')

        # Button to export cleaned data
        st.download_button(
            label="Export File",
            data=csv_data,
            file_name=filename,
            mime="text/csv"
        )