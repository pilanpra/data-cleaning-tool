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
    sanitized_columns = [re.sub(r'\W|^(?=\d)', '_', col) for col in df.columns]
    for old_col, new_col in zip(df.columns, sanitized_columns):
        df = df.withColumnRenamed(old_col, new_col)
    return df

def read_file(file):
    if file is not None:
        file_type = file.name.split('.')[-1]
        
        # Save file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}') as temp_file:
            temp_file.write(file.getvalue())
            temp_file_path = temp_file.name

        if file_type == 'csv':
            df = spark.read.csv(temp_file_path, header=True, inferSchema=True)
        elif file_type == 'json':
            df = spark.read.json(temp_file_path)
        elif file_type in ['xls', 'xlsx']:
            pd_df = pd.read_excel(temp_file_path)
            df = spark.createDataFrame(pd_df)
        else:
            st.error("Unsupported file type!")
            df = None
        
        # Clean up the temporary file
        os.remove(temp_file_path)

        # Sanitize column names
        if df is not None:
            df = sanitize_column_names(df)
        
        return df
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

file = st.file_uploader("Upload your file", type=['csv', 'json', 'xls', 'xlsx'])
df = read_file(file)

if df:
    st.write('\n')
    st.write('Data Overview:')
    st.write('\n')
    st.dataframe(df.toPandas())

st.markdown("# Data Summary")

if df:
    dataset_summary(df)