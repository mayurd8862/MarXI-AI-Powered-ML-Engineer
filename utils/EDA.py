import numpy as np
import pandas as pd
import streamlit as st
#from pandas_profiling import ProfileReport
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

def data_analysis(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        pr = ProfileReport(df, explorative=True)
        pr.to_file("Analysis.json")
        st.header('**Input DataFrame**')
        st.write(df)
        st.write('---')
        st.header('**Pandas Profiling Report**')
        st_profile_report(pr)
    else:
        st.info('Awaiting for CSV file to be uploaded.')    
