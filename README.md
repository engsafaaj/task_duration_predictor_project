IU International University of Applied Sciences Germany 
DLMDSPWP01 – PROGRAMMING WITH PYTHON

Predictive Modeling for Task Duration in Construction Projects
A Comprehensive Data-Driven Approach
# Task Duration Predictor

This project aims to predict the duration of tasks in various types of projects based on input features such as project type, size, duration, task name, and description.

## Project Structure

- `train_model.py`: Script to train the machine learning model.
- `main.py`: Script to run the graphical user interface (GUI) application for predicting task durations.
- `test_model.py`: Script to test the model with sample data and print the predicted task durations.
- `data.csv`: The dataset used for training and testing the model.
- `task_duration_predictor.joblib`: The trained machine learning model.
- `README.md`: Documentation of the project including setup instructions.

## Setup Instructions

### 1. Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/](https://github.com/engsafaaj/task_duration_predictor_project.git)
```

### 2. Install Dependencies

Ensure you have Python installed. Then, install the required libraries using pip:

```bash
pip install pandas scikit-learn joblib
```

### 3. Train the Model

Run the `train_model.py` script to train the model and save it as `task_duration_predictor.joblib`:

```bash
python train_model.py
```

### 4. Run the GUI Application

Run the `main.py` script to start the GUI application for predicting task durations:

```bash
python main.py
```

### 5. Test the Model

Run the `test_model.py` script to test the model with sample data and print the results:

```bash
python test_model.py
```

## Dataset Description

The dataset contains information about various tasks performed in different types of projects. The columns are as follows:

- **project_type**: Type of project (e.g., Infrastructure, Residential, Commercial, Industrial)
- **project_size**: Size of the project in square meters (sq m)
- **project_duration**: Total duration of the project in days
- **task_name**: Name of the task performed (e.g., Roofing, Framing, Plumbing, Electrical, Painting)
- **task_description**: Textual description of the task
- **task_duration**: Actual duration of the task in days (target variable)

## Tools and Libraries Used

- **Python**: Primary programming language for the project
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations and array handling
- **scikit-learn**: Machine learning model development and evaluation
  - **LinearRegression**: Regression model for predicting task duration
  - **OneHotEncoder**: Encoding categorical variables
  - **StandardScaler**: Scaling numerical variables
  - **TfidfVectorizer**: Transforming text data into numerical features
  - **train_test_split**: Splitting the dataset into training and testing sets
  - **mean_absolute_error, mean_squared_error, r2_score**: Model evaluation metrics
- **joblib**: Saving and loading the trained model
- **tkinter**: Creating the GUI for user interaction
- **ttk**: Enhanced themed widget set for tkinter
- **Jupyter Notebook**: Exploratory data analysis and model prototyping
- **IDE (VS Code, PyCharm)**: Writing and debugging Python scripts
## Acknowledgments

- This project was created with the help of scikit-learn, pandas, and tkinter libraries.
