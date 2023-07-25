# Importing libraries
import os 
import pandas as pd 
from constants import openai_key
import matplotlib.pyplot as plt 
import seaborn as sns 
from langchain.agents import create_pandas_dataframe_agent 
from langchain.llms import OpenAI 

import IPython
from langchain.llms import OpenAI
# import streamlit as st 

os.environ["OPENAI_API_KEY"] = openai_key

# Importing the data
df = pd.read_csv('http://www-stat.stanford.edu/~tibs/ElemStatLearn/datasets/SAheart.data') 

print(df.head(5))
# Initializing the agent 
agent = create_pandas_dataframe_agent(OpenAI(temperature=0), 
              df, verbose=True) 
openai = OpenAI(temperature=0.8) 
openai.model_name # This will print the model being used, 
                  # by default it uses ‘text-davinci-003’
# Let's check the shape of data.' 
agent("What is the shape of the dataset?") 
agent("How many rows and columns are present in the dataset?") 
# identifying missing values 
agent("How many missing values are there in each column?")
agent("Are there any empty value present in the dataset?")
# Let us see how the data looks like 
agent("Display {no} records in form of a table.")
# agent("Show the distribution of people suffering with chd using bar graph.")
agent("""Draw boxplot to find out if there are any outliers 
in terms of age of who are suffering from chd.""")
# Does Tobacco Cause CHD? 
agent("""validate the following hypothesis with t-test. 
Null Hypothesis: Consumption of Tobacco does not cause chd. 
Alternate Hypothesis: Consumption of Tobacco causes chd.""")
# How is the distribution of CHD across various age groups 
agent("""Plot the distribution of age for both the values 
of chd using kde plot. Also provide a lenged and 
label the x and y axises.""")