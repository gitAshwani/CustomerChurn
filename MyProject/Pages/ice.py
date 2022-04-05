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


def plot_ice1():
    
    fig1, ax1 = plt.subplots(figsize=(5, 3))
    ice = PartialDependenceDisplay.from_estimator(estimator=main.random_forest,
                            X=main.X_train,
                            features=[1,4,6],
                            target = 1,
                            kind="both",
                            ice_lines_kw={"color":"#808080","alpha": 0.3, "linewidth": 0.5},
                            pd_line_kw={"color": "#3498DB", "linewidth": 4, "alpha":1},                            
                            ax=ax1)
    fig1.tight_layout(pad=2.0)


    return fig1

def plot_ice2():
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    ice = PartialDependenceDisplay.from_estimator(estimator=main.random_forest,
                        X=main.X_train,
                        features=[9,14],
                        target = 1,
                        kind="both",
                        ice_lines_kw={"color":"#808080","alpha": 0.3, "linewidth": 0.5},
                        pd_line_kw={"color": "#ffa500", "linewidth": 4, "alpha":1},                        
                        ax=ax2)
    fig2.tight_layout(pad=2.0)
    
    return fig2

def plot_ice3():
    fig3, ax3 = plt.subplots(figsize=(5, 3))
    ice = PartialDependenceDisplay.from_estimator(estimator=main.random_forest,
                        X=main.X_train,
                        features=[16],
                        target = 1,
                        kind="both",
                        ice_lines_kw={"color":"#808080","alpha": 0.3, "linewidth": 0.5},
                        pd_line_kw={"color": "#ffa500", "linewidth": 4, "alpha":1},                    
                        ax=ax3)
    fig3.tight_layout(pad=2.0)
    
    

    return fig3






def run():
    st.title("ICE")

    st.pyplot(plot_ice1())
    st.pyplot(plot_ice2())
    st.pyplot(plot_ice3())

