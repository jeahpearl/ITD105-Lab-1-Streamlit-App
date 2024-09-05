import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
from scipy.stats.mstats import winsorize
from scipy.stats import ttest_ind

# Title of the app
st.title('Student Performance Data Dashboard')

# File uploader
uploaded_file = st.file_uploader("Upload CSV file here", type="csv")

if uploaded_file is not None:
    # 1. Load the data
    df = pd.read_csv(uploaded_file, delimiter=';')

    # 2. Sidebar filters for Gender and Parental Education
    st.sidebar.header("Filter the dataset")
    gender_filter = st.sidebar.selectbox('Filter by Gender', options=['All', 'M', 'F'], index=0)
    medu_filter = st.sidebar.selectbox('Filter by Mother\'s Education (Medu)', 
                                       options=['All'] + sorted(df['Medu'].unique().tolist()), index=0)
    fedu_filter = st.sidebar.selectbox('Filter by Father\'s Education (Fedu)', 
                                       options=['All'] + sorted(df['Fedu'].unique().tolist()), index=0)

    # Apply filters to the dataframe
    if gender_filter != 'All':
        df = df[df['sex'] == gender_filter]
    
    if medu_filter != 'All':
        df = df[df['Medu'] == medu_filter]
    
    if fedu_filter != 'All':
        df = df[df['Fedu'] == fedu_filter]

    # Tabs for different charts and visualizations
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
        ["Data Preview", "Summary Statistics", "Histograms", "Box Plots", "Heatmap", "Scatter Plot", "Gender Impact", "Pair Plot", "Winsorizing"]
    )

    # 3. Data Preview
    with tab1:
        st.subheader('Data Preview')
        st.write(df.head())

    # 4. Data Summary - Generate descriptive statistics
    with tab2:
        st.subheader('Summary Statistics')
        st.write(df.describe())

    # 5. Data Visualization - Histograms
    with tab3:
        st.subheader('Histograms')
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            fig, ax = plt.subplots()
            df[col].hist(ax=ax, bins=20)
            ax.set_title(f'Histogram of {col}')
            st.pyplot(fig)
            plt.close(fig)

    # Box and Whisker Plots
    with tab4:
        st.subheader('Box and Whisker Plots')
        for col in num_cols:
            fig, ax = plt.subplots()
            sns.boxplot(x=df[col], ax=ax)
            ax.set_title(f'Box and Whisker Plot of {col}')
            st.pyplot(fig)
            plt.close(fig)

    # Correlation Heatmap Including Gender
    with tab5:
        st.subheader('Improved Correlation Heatmap Including Gender')
        df['gender'] = df['sex'].map({'F': 0, 'M': 1})  # Convert 'sex' to numeric
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, annot_kws={"size": 10}, fmt='.2f', linewidths=0.5)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.title('Correlation Heatmap Including Gender', fontsize=14)
        st.pyplot(fig)
        plt.close(fig)

    # Interactive Scatter Plot using Plotly
    with tab6:
        st.subheader('Interactive Scatter Plot (Plotly)')
        col1 = st.selectbox('Select X-axis', num_cols)
        col2 = st.selectbox('Select Y-axis', num_cols, index=1)
        if col1 and col2:
            fig = px.scatter(df, x=col1, y=col2, title=f'Scatter Plot of {col1} vs {col2}')
            st.plotly_chart(fig)

    # Gender Impact Analysis on Final Exam Scores
    with tab7:
        st.subheader('Bar Chart of Gender Impact on Final Exam Scores')
        if 'sex' in df.columns and 'G3' in df.columns:
            gender_scores = df.groupby('sex')['G3'].mean()
            st.write('Average Final Exam Scores by Gender:', gender_scores)
            fig, ax = plt.subplots()
            sns.barplot(x=gender_scores.index, y=gender_scores.values)
            ax.set_title('Average Final Exam Scores by Gender')
            ax.set_ylabel('Average Score (G3)')
            st.pyplot(fig)
            plt.close(fig)

            # Perform t-test
            male_scores = df[df['sex'] == 'M']['G3']
            female_scores = df[df['sex'] == 'F']['G3']
            t_stat, p_val = ttest_ind(male_scores, female_scores)
            st.write(f"T-test results -- T-statistic: {t_stat}, P-value: {p_val}")

    # Pair Plot
    with tab8:
        st.subheader('Pair Plot of Selected Variables')
        if 'studytime' in df.columns and 'G1' in df.columns and 'G2' in df.columns:
            selected_variables = ['G1', 'G2', 'studytime']
            pair_plot_df = df[selected_variables].dropna()
            fig = sns.pairplot(pair_plot_df)
            st.pyplot(fig)
        else:
            st.error("The dataset doesn't contain the necessary columns for the pair plot. Please ensure it includes 'G1', 'G2', and 'studytime'.")

    # Winsorizing and Summary Statistics
    with tab9:
        st.subheader('Winsorizing')
        df_winsorized = df.copy()
        for col in num_cols:
            df_winsorized[col] = winsorize(df_winsorized[col], limits=[0.05, 0.05])
        st.write('Data after winsorizing:', df_winsorized.head())

        # Data Summary after winsorizing
        st.subheader('Summary Statistics (Winsorized Data)')
        st.write(df_winsorized.describe())

        # Data Information after winsorizing
        st.subheader('Data Info (Winsorized Data)')
        buffer = io.StringIO()
        df_winsorized.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

else:
    st.warning('Please upload a CSV file to analyze.')
