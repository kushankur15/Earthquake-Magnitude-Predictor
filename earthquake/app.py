import streamlit as st
import pandas as pd
import plotly.express as px
from model import predict_magnitude
import matplotlib.pyplot as plt

st.title("Earthquake Magnitude Predictor")

df = pd.read_csv("earthquake.csv")
df.dropna(subset=['mag'], inplace=True)
df['magError'].fillna(df['magError'].mean(), inplace=True)
df['magNst'].fillna(df['magNst'].mean(), inplace=True)
df['horizontalError'].fillna(df['horizontalError'].mean(), inplace=True)
df['nst'].fillna(df['nst'].mean(), inplace=True)
df['gap'].fillna(df['gap'].mean(), inplace=True)
df['dmin'].fillna(df['dmin'].mean(), inplace=True)


if st.button("View Data"):
    df = pd.read_csv("earthquake.csv")
    st.dataframe(df)

elif st.button("View Visualization"):
    fig1 = px.scatter_geo(df,
                     lat='latitude',
                     lon='longitude',
                     color='mag',
                     hover_name='place',
                     projection='natural earth',
                     title='Global Earthquakes',
                     color_continuous_scale='purd')
    fig1.update_layout(legend=dict(title='Magnitude'))
    st.plotly_chart(fig1)

    magtype = df.groupby('magType')['mag'].mean().reset_index()
    fig2 = px.pie(magtype,
             values=df['magType'].value_counts().values,
             names='magType',
             title='Average Magnitude by Type',
             color_discrete_sequence=px.colors.sequential.Pinkyl_r)
    st.plotly_chart(fig2)

    fig = px.bar(x=magtype['magType'], y=magtype['mag'], title='Average Magnitude by Type')
    fig.update_layout(xaxis_title='Magnitude Type', yaxis_title='Average Magnitude')
    st.plotly_chart(fig)

    fig4 = px.pie(values=magtype['mag'].values,
             names=magtype['magType'].values,
             title='Magnitude Source Distribution',
             color_discrete_sequence=px.colors.sequential.Plasma_r)
    fig4.update_layout(xaxis_title='Magnitude Source', yaxis_title='Average Magnitude')
    st.plotly_chart(fig4)

    fig5 = px.scatter(df, x='horizontalError', y='mag', color='status', title='Horizontal Error vs Magnitude by Status')
    st.plotly_chart(fig5)

    fig6 = px.scatter(df, x='dmin', y='mag', color='status', title='Minimum Distance vs Magnitude by Status')
    st.plotly_chart(fig6)

    fig7 = px.scatter(df, x='rms', y='mag', color='status', title='RMS vs Magnitude by Status')
    st.plotly_chart(fig7)

latitude = st.number_input("Latitude", value=0.0)
longitude = st.number_input("Longitude", value=0.0)
depth = st.number_input("Depth", value=0.0)
nst = st.number_input("NST", value=0)
gap = st.number_input("Gap", value=0.0)
dmin = st.number_input("DMin", value=0.0)
rms = st.number_input("RMS", value=0.0)
horizontalError = st.number_input("Horizontal Error", value=0.0)
depthError = st.number_input("Depth Error", value=0.0)
net = st.selectbox("Net", ['nc', 'ci', 'ak', 'tx', 'av', 'us', 'uu', 'uw', 'nn', 'ok', 'mb', 'pr','hv', 'se', 'nm'])
status = st.selectbox("Status", ['reviewed', 'automatic'])
locationSource = st.selectbox("Location Source", ['nc', 'ci', 'ak', 'tx', 'av', 'us', 'uu', 'uw', 'nn', 'ok', 'mb', 'pr','hv', 'se', 'nm'])

if st.button("Predict"):
    new_eq = {
        "latitude": latitude,
        "longitude": longitude,
        "depth": depth,
        "nst": nst,
        "gap": gap,
        "dmin": dmin,
        "rms": rms,
        "horizontalError": horizontalError,
        "depthError": depthError,
        "net": net,
        "status": status,
        "locationSource": locationSource
    }
    st.write("Predicted Magnitude:", predict_magnitude(new_eq))

