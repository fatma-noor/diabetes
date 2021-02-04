import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.model_selection import train_test_split
#import xgboost
from sklearn.ensemble import RandomForestClassifier

st.title('Automated Diabetes checking system')
st.markdown("""
	This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

The datasets consists of several medical predictor variables and one target variable, Outcome. Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.

Columns

1.Pregnancies = Number of times pregnant

2.Glucose = Plasma glucose concentration a 2 hours in an oral glucose tolerance test

3.BloodPressure = Diastolic blood pressure (mm Hg)

4.SkinThickness = Triceps skin fold thickness (mm)

5.Insulin = 2-Hour serum insulin (mu U/ml)

6.BMI = Body mass index (weight in kg/(height in m)^2)

7.DiabetesPedigreeFunction = Diabetes pedigree function

8.Age = Age (years)

9.Outcome = Class variable (0 or 1) 268 of 768 are 1, the others are 0

	""")
#read the csv file
df=pd.read_csv('diabetes.csv')
diabetes_map = {True: 1, False: 0}
df['diabetes']= df['diabetes'].map(diabetes_map)
x=df.drop(['diabetes'], axis= 1)
y=df.diabetes.values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state= 0)

def predict(num_preg, diab_pred, age, glucose_conc, skin, thickness, insulin, bmi, diastolic_bp):
	inp = list([num_preg, diab_pred, age, glucose_conc, skin, thickness, insulin, bmi, diastolic_bp])
	X = pd.DataFrame([inp], columns=[num_preg, diab_pred, age, glucose_conc, skin, thickness, insulin, bmi, diastolic_bp])
	st.write(X)
	scaler = StandardScaler().fit(x)
	transformed_X =  scaler.transform(X)
	st.write(transformed_X)
	
	rf = RandomForestClassifier(n_estimators=500, min_samples_split=6, min_samples_leaf=2, max_features='sqrt', max_depth=5)
	rf.fit(X_train, y_train)
	print("Random Forest accuracy: {: .2f}%".format(rf.score(X_test, y_test)*100))
	y_pred = rf.predict(transformed_X)
	return y_pred



num_preg = st.number_input('Numer of Pregnancies')
glucose_conc = st.number_input('Plasma glucose concentration a 2 hours in an oral glucose tolerance test')

diastolic_bp = st.number_input('Diastolic blood pressure (mm Hg)')

thickness = st.number_input('Triceps skin fold thickness (mm)')

insulin = st.number_input('2-Hour serum insulin (mu U/ml)')

bmi = st.number_input('Body mass index (weight in kg/(height in m)^2)')

diab_pred = st.number_input("Diabetes pedigree function")

age = st.number_input('Age (years)')
skin = st.number_input('skin')


if st.button('check diabetes status'):
	pred_value =predict(num_preg, diab_pred, age, glucose_conc, skin, thickness, insulin, bmi, diastolic_bp) 
	if pred_value == 1:
		st.success('The patient is fine')

	else:
		st.warning('The patient has diabetes')