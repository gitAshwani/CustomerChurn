import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import shap
from Pages import main
shap.initjs()



# Exact Explainer (.predict)
def run_exact():
    global explainer, shap_values
    explainer = explainer = shap.Explainer(main.random_forest.predict, main.X_train)
    shap_values = explainer(main.X_test)


def run_kernel():
    global explainer, shap_values
    explainer = shap.KernelExplainer(main.random_forest.predict_proba, main.X_train)
    shap_values = explainer.shap_values(main.X_test)

run_exact()

# helper to plot js plots

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# force plot (requires js helper)

def plot_force():
    fig = shap.force_plot(explainer.expected_value[1], shap_values[1], main.X_test)
    return fig

# 

def plot_bar():
    fig = plt.figure()
    shap.plots.bar(shap_values, show=False)
    return fig

def plot_beeswarm():
    fig = plt.figure()
    shap.plots.beeswarm(shap_values)
    return fig

def plot_cohorts():
    fig = plt.figure()
    shap.plots.bar(shap_values.cohorts(2).abs.mean(0))
    return fig

def run():
    st.title("SHAP")

    selectbox = st.sidebar.selectbox(
        "Select SHAP Explainer type:",
        ("ExactExplainer", "KernelExplainer"))

    if selectbox =="ExactExplainer":
  #       run_exact()
        c1,c2 = st.columns(2)
        c1.pyplot(plot_bar())
        c2.pyplot(plot_cohorts())
        st.pyplot(plot_beeswarm())
        
    elif selectbox =="KernelExplainer":
        run_kernel()

        # js plot
        st_shap(plot_force(), 400)



        

    



