
# Calorie Burnt Prediction - Machine Learning Project

This project predicts the number of calories burned during exercise based on various physiological parameters such as age, gender, weight, and other exercise-related details. The machine learning model is built using a Random Forest Regressor, and the dataset includes user-specific exercise data and corresponding calorie burn values.

## Project Overview

This repository contains three key files:
1. **`calorie_webapp.py`**: Script for loading a pre-trained model and making predictions on user input.
2. **`prediction_system.py`**: Script to predict calorie burn using a pre-trained model with custom input data.
3. **`calorie_burnt.ipynb`**: Jupyter Notebook containing the full process of data preparation, training, evaluation, and saving the Random Forest model.

## Table of Contents

1. [Installation](#installation)
2. [Dataset Overview](#dataset-overview)
3. [Project Structure](#project-structure)
4. [Model Training](#model-training)
5. [How to Use](#how-to-use)
6. [Technologies Used](#technologies-used)
7. [Contributing](#contributing)
8. [License](#license)

---

## Installation

To get started, clone this repository:

```bash
git clone https://github.com/your-username/calorie-burn-prediction.git
cd calorie-burn-prediction
```

### Dependencies

Make sure you have the following libraries installed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

These libraries are essential for data manipulation, visualization, and building the machine learning model.

---

## Dataset Overview

The dataset used in this project consists of two CSV files:
- **`calories.csv`**: Contains calorie burn information.
- **`exercise.csv`**: Contains user details such as gender, age, height, weight, and exercise data (duration, heart rate, body temperature).

### Sample Data

`calories.csv`:
| User_ID  | Calories |
|----------|----------|
| 14733363 | 231.0    |
| 14861698 | 66.0     |
| 11179863 | 26.0     |

`exercise.csv`:
| User_ID  | Gender | Age | Height | Weight | Duration | Heart_Rate | Body_Temp |
|----------|--------|-----|--------|--------|----------|------------|-----------|
| 14733363 | male   | 68  | 190.0  | 94.0   | 29.0     | 105.0      | 40.8      |
| 14861698 | female | 20  | 166.0  | 60.0   | 14.0     | 94.0       | 40.3      |

---

## Project Structure

- **`calorie_webapp.py`**: Loads the pre-trained model and predicts calorie burn based on user input.
- **`prediction_system.py`**: Provides a system for making predictions using a saved model.
- **`calorie_burnt.ipynb`**: Contains the entire process from data loading, cleaning, training, testing, and saving the model.

### Key Steps in `calorie_burnt.ipynb`:

1. **Data Loading**: Load the `calories.csv` and `exercise.csv` datasets.
2. **Data Cleaning**: 
   - Merge both datasets on `User_ID`.
   - Handle missing values (if any).
   - Encode categorical data (e.g., Gender).
3. **Exploratory Data Analysis**:
   - Use visualizations like count plots and distribution plots to understand data trends.
   - Calculate the correlation between features using a heatmap.
4. **Model Training**:
   - Split the data into training and testing sets (80-20 split).
   - Train a Random Forest Regressor model on the training data.
   - Evaluate model performance using Mean Absolute Error (MAE).
5. **Saving the Model**: Save the trained model using `pickle` for later use in prediction systems.

---

## Model Training

In the notebook (`calorie_burnt.ipynb`), the data is used to train a Random Forest Regressor. The key steps include:
1. **Feature Selection**: Features such as `Gender`, `Age`, `Height`, `Weight`, `Duration`, `Heart_Rate`, and `Body_Temp` are selected as input features. The target variable is `Calories`.
2. **Model Training**:
   - `RandomForestRegressor(n_estimators=10)` is used to train the model.
3. **Model Evaluation**:
   - The model performance is evaluated using the Mean Absolute Error (MAE), achieving an MAE of approximately 2.03.

---

## How to Use

1. **Running the Web App**:
   Use the `calorie_webapp.py` script to load the pre-trained model and predict calories based on user input.

   Example usage:
   ```python
   import numpy as np
   import pickle

   # Load the saved model
   loaded_model = pickle.load(open('trainedcalmodel.sav', 'rb'))

   # Input data (Gender, Age, Height, Weight, Duration, Heart Rate, Body Temp)
   input_data = (1, 20, 166.0, 60.0, 14.0, 94.0, 40.3)

   # Reshape and predict
   input_data_as_array = np.asarray(input_data).reshape(1, -1)
   prediction = loaded_model.predict(input_data_as_array)

   print("Predicted Calories Burned: ", prediction)
   ```

2. **Using the Model for Prediction**:
   Run `prediction_system.py` with any user-specific data to predict the number of calories burned.

---

## Technologies Used

- **Languages**: Python
- **Libraries**: 
  - `NumPy`
  - `Pandas`
  - `Matplotlib` and `Seaborn` for visualizations
  - `Scikit-learn` for model building and evaluation
- **Machine Learning Algorithm**: Random Forest Regressor

---

## Contributing

If you'd like to contribute to this project, feel free to fork the repository and submit a pull request. Please ensure that your changes are consistent with the current code structure and project goals.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
