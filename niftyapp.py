import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.linear_model import Lasso

# Load data
df = pd.read_csv('cleaned_nifty.csv')

# Extract years and annual returns
years = df['Year']
annual_returns = df['Annual']

# Split the data into features and target variable
X = df.drop(columns=['Year', 'Annual'])
y = df['Annual']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Grid search parameters
param_grid = {'model__alpha': [0.1, 1.0, 10.0]}

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Lasso())
])

# Grid search
grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=0)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Streamlit app
def main():
    st.title('Annual Nifty Prediction App')

    app_mode = st.sidebar.selectbox("Choose the mode", ["Input Monthly Returns", "Plot Annual Returns"], key='selectbox')

    if app_mode == "Input Monthly Returns":
        st.header('Input Monthly Returns')
        st.markdown('Enter the monthly returns to predict the annual return.')

        # Sidebar for user input
        jan = st.number_input('January', value=0.0)
        feb = st.number_input('February', value=0.0)
        mar = st.number_input('March', value=0.0)
        apr = st.number_input('April', value=0.0)
        may = st.number_input('May', value=0.0)
        jun = st.number_input('June', value=0.0)
        jul = st.number_input('July', value=0.0)
        aug = st.number_input('August', value=0.0)
        sep = st.number_input('September', value=0.0)
        oct_ = st.number_input('October', value=0.0)
        nov = st.number_input('November', value=0.0)
        dec = st.number_input('December', value=0.0)

        if st.button('Predict'):
            # Predict annual value
            input_data = pd.DataFrame({
                'Jan': [jan], 'Feb': [feb], 'Mar': [mar], 'Apr': [apr], 'May': [may], 'Jun': [jun],
                'Jul': [jul], 'Aug': [aug], 'Sep': [sep], 'Oct': [oct_], 'Nov': [nov], 'Dec': [dec]
            })
            annual_prediction = best_model.predict(input_data)

            st.header('Annual Prediction')
            st.write(f'The predicted annual value is: {annual_prediction[0]}')

    elif app_mode == "Plot Annual Returns":
        st.header('Annual Returns Over Time')
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.plot(years, annual_returns, marker='o', linestyle='-', color='b')
        ax.set_title('Annual Returns Over Time')
        ax.set_xlabel('Year')
        ax.set_ylabel('Annual Returns')
        ax.grid(True)
        plt.tight_layout()
        st.pyplot(fig)

    # Custom CSS to make selectbox larger and more prominent
    st.markdown(
        """
        <style>
        .css-1a96jnm-option {
            font-size: 18px !important;
            font-weight: bold !important;
            padding: 12px 20px !important;
            margin-right: 10px !important;
            border-radius: 10px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
