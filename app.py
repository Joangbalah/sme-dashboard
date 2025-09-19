import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title("SME Cash Flow Dashboard")

# Upload a CSV file
uploaded_file = st.file_uploader("Upload your SME data (CSV format)", type="csv")

if uploaded_file is not None:
    # Read CSV
    data = pd.read_csv(uploaded_file, parse_dates=['Date'])
    
    # Calculate Cash Flow
    data['CashFlow'] = data['Income'] - data['Expense']
    data['Month'] = data['Date'].dt.month
    
    # Show data
    st.subheader("Historical Data")
    st.dataframe(data)
    
    # Train Linear Regression Model
    X = data[['Month']]
    y = data['CashFlow']
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict next 3 months
    future_months = pd.DataFrame({'Month': [max(X['Month'])+1,
                                            max(X['Month'])+2,
                                            max(X['Month'])+3]})
    predictions = model.predict(future_months)
    
    st.subheader("Predicted Cash Flow (Next 3 Months)")
    pred_df = pd.DataFrame({
        'Month': future_months['Month'],
        'Predicted CashFlow': predictions
    })
    st.dataframe(pred_df)
    
    # Plot
    plt.plot(data['Month'], y, marker='o', label='Historical')
    plt.plot(future_months['Month'], predictions, marker='x',
             linestyle='--', color='orange', label='Predicted')
    plt.xlabel('Month')
    plt.ylabel('Cash Flow ($)')
    plt.title('SME Cash Flow Forecast')
    plt.legend()
    st.pyplot(plt)
