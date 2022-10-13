import streamlit as st
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import os

st.set_page_config(layout="wide")
state = st.session_state

if "init" not in state:
    
    state.init = True
    state.weights = {}
    state.config = ""
    state.result = None
    state.result_dict = {}
    state.logs = ""

    state.url = url="http://localhost:8080/start"
    if os.path.isfile("/url.txt"):
        with open("/url.txt", "r") as f:
            state.url = f.readline().strip() + 'start'

col1, col2 = st.columns(2)


with col1:
    json_text = st.text_area("Paste the configuration JSON", key='json_text')
with col2:
    config_file = st.file_uploader("Choose a configuration file", ["json"], accept_multiple_files=False)

if json_text:
    state.config = json_text
if config_file:
    state.config = config_file.read().decode('UTF-8')

if state.config:
    starter = st.button("Run Config")
    if starter:
        with st.spinner('calculating'):
            headers = {
                'Content-Type': 'application/json'
                }
            payload = state.config
            
            response = requests.request("POST", state.url, headers=headers, data=payload)
            state.result = response.json()
            state.logs = state.result['logs']
            state.result_dict = json.loads(state.result['result'])['result']
            state.weights = state.result_dict.get('trained_model', {})
            st.success('Done!')
if state.result:

    #confusion matrix
    tp = state.result_dict.get("TP")
    fp = state.result_dict.get("FP")
    fn = state.result_dict.get("FN")
    tn = state.result_dict.get("TN")
    if tp is not None and fp is not None and fn is not None and tn is not None:
        confusion =[[tp, fp], [fn, tn]]

        df = pd.DataFrame(confusion)
        fig, ax = plt.subplots()
        sn.heatmap(df, annot=True, fmt='g', xticklabels= [1, 0], yticklabels=[1, 0], ax=ax)

        ax.xaxis.tick_top()
        ax.set_xlabel('Predicted')    
        ax.xaxis.set_label_position('top')
        ax.set_ylabel('Actual')

        col1, col2 = st.columns([3,3])
        with col1:
            st.pyplot(fig)
            mcc = state.result_dict.get("MCC")
            if mcc is not None:
                st.text(f"The Matthews Correlation Coefficient is {mcc:.3f}")

    col1, col2 = st.columns([8,2])
    with col1:
        with st.expander("JSON Result"):
            st.json(state.result_dict)
    with col2:
        st.download_button("Download JSON", data=json.dumps(state.result_dict), file_name="result.json")
    
    if state.weights:
        col1, col2 = st.columns([8,2])
        with col1:
            with st.expander("Network Weights"):
                st.json(state.weights)
        with col2:
            st.download_button("Download Network Weights", data=json.dumps(state.weights), file_name="weights.json")
