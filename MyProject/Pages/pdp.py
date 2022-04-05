import streamlit as st

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from Pages import main
from sklearn.inspection import plot_partial_dependence


fig, ax = plt.subplots(figsize=(12, 8))
PartialDependenceDisplay.from_estimator(main.random_forest, main.X_train, [1,2,(1,2),(2,3),(3,4),(4,5)], target=1,ax=ax)
fig.tight_layout(pad=2.0)


def plot_pdp():
    fig, ax = plt.subplots(figsize=(12, 8))
    PartialDependenceDisplay.from_estimator(main.random_forest, main.X_train, [1,2,(1,2),(2,3),(3,4),(4,5)], target=1,ax=ax)
    fig.tight_layout(pad=2.0)
    return fig

def run():
    st.title("PDP")
    
    st.pyplot(plot_pdp())
