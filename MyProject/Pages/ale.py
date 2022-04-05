import streamlit as st

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from Pages import main
import PyALE

def ale(target=None, print_meanres=False, **kwargs):
    if target is not None:
        class clf():
            def __init__(self, classifier):
                self.classifier = classifier
            def predict(self, X):
                return(self.classifier.predict_proba(X)[:, target])
        clf_dummy = clf(kwargs["model"])
        kwargs["model"] = clf_dummy
    if (print_meanres & len(kwargs["feature"])==1):
        mean_response = np.mean(kwargs["model"].predict(kwargs["X"]), axis=0)
        print(f"Mean response: {mean_response:.5f}")
    return PyALE.ale(**kwargs)


def plot_ale1d():
    fig, axs = plt.subplots(3,2,figsize=(10, 10), sharey=True)
    fig.tight_layout(pad=3)
    
    # ALE plot 1D - selected features, target=1 (churn)
    ale_number_customer_service_calls = ale(
    X=main.X_train,
    model=main.random_forest,
    feature= ["number_customer_service_calls"],
    include_CI=True,
    target=1,
    fig = fig,
    ax=axs[0, 0],
    print_meanres=True)
    
    ale_total_day_charge = ale(
    X=main.X_train,
    model=main.random_forest,
    feature= ["total_day_charge"],
    include_CI=True,
    target=1,
    fig = fig,
    ax=axs[0,1],
    print_meanres=True
)
    ale_total_day_minutes = ale(
    X=main.X_train,
    model=main.random_forest,
    feature= ["total_day_minutes"],
    include_CI=True,
    target=1,
    fig = fig,
    ax=axs[1,0],
    print_meanres=True)
    
    ale_total_intl_calls = ale(
    X=main.X_train,
    model=main.random_forest,
    feature= ["total_intl_calls"],
    include_CI=True,
    target=1,
    fig = fig,
    ax=axs[1,1],
    print_meanres=True
)


    ale_international_plan = ale(
    X=main.X_train,
    model=main.random_forest,
    feature= ["international_plan"],
    include_CI=True,
    target=1,
    fig = fig,
    ax=axs[2,0],
    print_meanres=True
)
    
    ale_total_eve_charge = ale(
    X=main.X_train,
    model=main.random_forest,
    feature= ["total_eve_charge"],
    include_CI=True,
    target=1,
    fig = fig,
    ax=axs[2,1],
    print_meanres=True
)

    
    return fig

def plot_ale2d(f1,f2):
    fig, ax = plt.subplots(figsize=(2, 2))
    ale_2d = ale(
        X=main.X_train,
        model=main.random_forest,
        feature=[f1,f2],
        include_CI=True,
        target=1, 
        fig=fig,
        ax=ax)
    return fig
    
def run():
    st.title("ALE")
    st.pyplot(plot_ale1d())
    st.pyplot(plot_ale2d("international_plan","total_day_charge"))
    st.pyplot(plot_ale2d("number_customer_service_calls","total_day_minutes"))
    st.pyplot(plot_ale2d("international_plan","total_intl_calls"))



