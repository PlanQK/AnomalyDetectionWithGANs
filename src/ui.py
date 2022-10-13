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
    #state.logs = ""
    state.file_id = None

    state.url = url="http://localhost:8080/start"
    if os.path.isfile("/url.txt"):
        with open("/url.txt", "r") as f:
            state.url = f.readline().strip() + 'start'


def load_config(path):
    with open(path, "r") as f:
        return f.read()

option = st.selectbox(
    'Example Configurations:',
    ('Bars and Stripes 2x2 Train', 'Bars and Stripes 2x2 Predict', 'Bars and Stripes 2x2 Unsupervised Train', 'Bars and Stripes 2x2 Unsupervised Predict',))
load_example = st.button("Load Example Config")

if load_example:
    name_to_filename = {
        'Bars and Stripes 2x2 Train':'2x2_training_classical.json',
        'Bars and Stripes 2x2 Predict':'2x2_prediction_classical.json',
        'Bars and Stripes 2x2 Unsupervised Train':'2x2_unsupervised_training_classical.json',
        'Bars and Stripes 2x2 Unsupervised Predict':'2x2_unsupervised_prediction_classical.json',
    }
    state.config = load_config('input/' + name_to_filename[option])

config_file_uploader = st.empty()
config_file = config_file_uploader.file_uploader("Choose a configuration file", ["json"], accept_multiple_files=False)


if config_file:
    if config_file.id != state.file_id:
        state.file_id = config_file.id
        state.config = config_file.read().decode('UTF-8')
    
else:
    state.file_id = None


split_config = st.checkbox('Split Config')

if split_config:
    try:
        config_dic = json.loads(state.config)
        col1, col2, col3 = st.columns([1,1,1])

        weights = config_dic.get('params', {}).pop('trained_model', {})

        with col1:
            st.text('Config:')
            st.json(config_dic.get('params', {}))
        with col2:
            st.text('Data:')
            st.json(config_dic.get('data', {}))
        with col3:
            st.text('Weights:')
            st.json(weights)
    except ValueError:
        st.warning('config is not a valid json, can not split')
        split_config = False
    
    
    
    
    
    
    
    

    

if not split_config:
    json_text = st.text_area("Paste the configuration JSON", value=state.config)

    if json_text:
        state.config = json_text

if state.config:
    starter = st.button("Run Config")
    if starter:
        with st.spinner('calculating'):
            headers = {
                'Content-Type': 'application/json'
                }
            payload = state.config
            
            
            response = requests.request("POST", state.url, headers=headers, data=payload)
            try:
                state.result = response.json()
                #state.logs = state.result['logs']
                state.result_dict = json.loads(state.result['result'])['result']
                state.weights = state.result_dict.get('trained_model', {})
                st.success('Done!')
            except:
                st.warning('malformed server response:\n' + response.text)
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
    

    # col1, col2 = st.columns([8,2])
    # with col1:
    #     with st.expander("Server Error Logs"):
    #         st.json(state.logs)
    # with col2:
    #     st.download_button("Download Server Error Logs", data=state.logs, file_name="logs.txt")
