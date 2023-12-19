import numpy as np
import pickle
import streamlit as st


# loading saved model

loaded_model = pickle.load(open('C:/Users/Owner/PycharmProjects/Calorie_Webapp/trainedcalmodel.sav', 'rb'))

# Creating function for prediction


def calorie_prediction(input_data):

    input_data_as_array = np.asarray(input_data)

    input_data_reshape = input_data_as_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshape)

    print(prediction)
    # print("Calorie burned Prediction : ", prediction)
    return prediction


def main():

    # giving title
    st.title('Calorie Prediction Webapp')

    # getting input  data from user
    # Gender,Age,Height,Weight,Duration,Heart_Rate,Body_Temp
    # male = 0  & female 1
    Gender = st.text_input('Gender Selection')
    Age = st.text_input('Age of person')
    Height = st.text_input('Height Selection')
    Weight = st.text_input('Weight Selection')
    Duration = st.text_input('Duration Selection')
    Heart_Rate = st.text_input('Heart Rate Selection')
    Body_Temp = st.text_input('Body Temperature Selection')

    # code for Prediction

    calorie_predicted = ''

    # creating a button for Prediction

    if st.button('Calorie Prediction'):
        calorie_predicted = calorie_prediction([Gender,Age, Height, Weight, Duration, Heart_Rate, Body_Temp])

        st.success(calorie_predicted)


if __name__ == '__main__':
    main()



