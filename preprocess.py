import pandas as pd 

data=pd.read_csv("diabetes.csv")

print(data.isnull().sum())

data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']]=data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,pd.NA)    

data.fillna(data.median(), inplace=True)


X=data.drop('Outcome', axis=1)
Y=data['Outcome']

# Assuming your cleaned dataframe is called df
data.to_csv(r"F:\DigitalTwin\processed_data.csv", index=False)
print("Cleaned CSV saved as processed_data.csv in F:\\DigitalTwin\\")


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)   


from sklearn.linear_model import LogisticRegression
model=LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report
y_pred=model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))


import joblib
joblib.dump(model, 'diabetes_model.pkl')