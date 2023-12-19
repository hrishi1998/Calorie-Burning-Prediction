import numpy as np
import pickle

#loading saved model

loaded_model = pickle.load(open('C:/Users/Owner/PycharmProjects/Calorie_Webapp/trainedcalmodel.sav', 'rb'))

input_data = (1, 20, 166.0, 60.0, 14.0, 94.0, 40.3)

input_data_as_array = np.asarray(input_data)

input_data_reshape = input_data_as_array.reshape(1, -1)

prediction = loaded_model.predict(input_data_reshape)

print(prediction)
