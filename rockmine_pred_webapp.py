import numpy as np
import pandas as pd
import pickle
import streamlit as st
# loading the saved model
import os
model_path = os.path.join(os.path.dirname(__file__), "trained_model.sav")
loaded_model = pickle.load(open(model_path, "rb"))
# creatinga function for prediction
def main():

    st.title("Rock vs Mine Prediction Web App")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        # Read uploaded CSV
        data = pd.read_csv(uploaded_file, header=None)

        st.subheader("Uploaded Data")
        st.dataframe(data)

        # If CSV has 60 feature columns
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

        # If CSV has 61 columns (last column is label)
        elif data.shape[1] == 61:

            st.info("Detected 61 columns. Removing the last label column.")

            X = data.iloc[:, :-1]

            if st.button("Predict"):

                prediction = loaded_model.predict(X)

                result = data.copy()
                result["Prediction"] = [
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