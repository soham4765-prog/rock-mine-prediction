import numpy as np
import pandas as pd
import pickle
import streamlit as st
# loading the saved model
import os
model_path = os.path.join(os.path.dirname(__file__), "trained_model.sav")
loaded_model = pickle.load(open(model_path, "rb"))
# creatinga function for prediction
def rock_mine_pred(input_data):
    #changing the input data to a numpy array
    input_data_as_numpy_array=np.asarray(input_data)
    #reshape the numpy array
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)
    if (prediction[0]=='R'):
        return "the object is rock"
    else :
        return "the object is mine"
    
def main():
    # title for web page
    st.title('rock vs mine prediction web app')
    # geting the input data from the user
    # Upload CSV
    uploaded_file = st.file_uploader("Choose a CSV file",type=["csv"])
    if uploaded_file is not None:
        # Read CSV
        data = pd.read_csv(uploaded_file, header=None)
        st.subheader("Uploaded Data")
        st.dataframe(data)
    # Check number of columns
    if data.shape[1] == 60:
        if st.button("Predict"):
            prediction = loaded_model.predict(data)
            result = pd.DataFrame({
                "Prediction": [
                    "Rock" if p == "R" else "Mine"
                    for p in prediction
                ]
            })
            st.success("Prediction Completed!")
            st.subheader("Prediction Results")
            st.dataframe(result)
    elif data.shape[1] == 61:
        st.info("Detected 61 columns. Assuming the last column is the label and removing it.")
        X = data.iloc[:, :-1]
        if st.button("Predict"):
            prediction = loaded_model.predict(X)
            result = data.copy()
            result["Predicted"] = [
                "Rock" if p == "R" else "Mine"
                for p in prediction
            ]
            st.success("Prediction Completed!")
            st.subheader("Prediction Results")
            st.dataframe(result)
    else:
        st.error(
            f"Invalid CSV! Expected 60 or 61 columns, but found {data.shape[1]} columns."
        )
if __name__ == "__main__":
    main()