import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Initialize empty lists to store user-provided sales data
dates = []
sales = []

def add_data():
    global date_entry, sales_entry, status_label

    user_date_str = date_entry.get().strip()
    user_sales_str = sales_entry.get().strip()

    if not user_date_str or not user_sales_str:
        messagebox.showwarning("Input Error", "Please enter a date and sales value.")
        return

    try:
        user_date = pd.to_datetime(user_date_str, format='%Y-%m-%d')
        user_sales = float(user_sales_str)

        dates.append(user_date)
        sales.append(user_sales)

        date_entry.delete(0, tk.END)
        sales_entry.delete(0, tk.END)

        status_label.config(text="Data added successfully.")
    except ValueError:
        messagebox.showerror("Input Error", "Invalid date or sales value. Please enter valid data.")

def predict_sales():
    global future_date_entry, prediction_label

    if len(dates) < 2:
        messagebox.showerror("Prediction Error", "Insufficient data for prediction. Please provide at least two data points.")
        return

    # Convert user-provided data to DataFrame
    sales_data = pd.DataFrame({'date': dates, 'sales': sales})

    # Sort the data by date
    sales_data.sort_values('date', inplace=True)

    # Initialize the Linear Regression model
    model = LinearRegression()

    # Train the model using the provided data
    model.fit(sales_data['date'].values.astype('int64').reshape(-1, 1), sales_data['sales'].values)

    # Get user input for the future date
    user_date_str = future_date_entry.get().strip()

    try:
        # Convert the user input date to pandas datetime type
        user_date = pd.to_datetime(user_date_str, format='%Y-%m-%d')

        # Make the prediction
        predicted_sales = model.predict([[user_date.timestamp()]])
        prediction_label.config(text=f"Predicted sales for {user_date_str}: {predicted_sales[0]}")

        # Create a DataFrame for predicted data for plotting
        predicted_data = pd.DataFrame({'date': [user_date], 'sales': [predicted_sales[0]]})

        # Plot historical sales data and predicted sales
        plt.figure(figsize=(10, 6))
        plt.plot(sales_data['date'], sales_data['sales'], marker='o', linestyle='-', label='Historical Sales')
        plt.plot(predicted_data['date'], predicted_data['sales'], marker='o', linestyle='-', color='red', label='Predicted Sales')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.title('Historical Sales Data and Sales Prediction')
        plt.legend()
        plt.show()

        # Generate a pie chart for sales distribution
        plt.figure(figsize=(8, 8))
        plt.pie(sales_data['sales'], labels=sales_data['date'].dt.strftime('%Y-%m-%d'), autopct='%1.1f%%', startangle=140)
        plt.axis('equal')
        plt.title('Sales Distribution')
        plt.show()

    except ValueError:
        messagebox.showerror("Input Error", "Invalid date format. Please use YYYY-MM-DD.")

# Create GUI
root = tk.Tk()
root.title("Sales Predictor")

label = tk.Label(root, text="Sales Predictor", font=("Helvetica", 16))
label.pack(pady=10)

date_label = tk.Label(root, text="Enter Date (YYYY-MM-DD):")
date_label.pack()

date_entry = tk.Entry(root)
date_entry.pack()

sales_label = tk.Label(root, text="Enter Sales Value:")
sales_label.pack()

sales_entry = tk.Entry(root)
sales_entry.pack()

add_button = tk.Button(root, text="Add Data", command=add_data)
add_button.pack(pady=10)

status_label = tk.Label(root, text="", font=("Helvetica", 12))
status_label.pack()

future_date_label = tk.Label(root, text="Enter Future Date (YYYY-MM-DD):")
future_date_label.pack()

future_date_entry = tk.Entry(root)
future_date_entry.pack()

predict_button = tk.Button(root, text="Predict Sales", command=predict_sales)
predict_button.pack(pady=10)

prediction_label = tk.Label(root, text="", font=("Helvetica", 12))
prediction_label.pack()

root.mainloop()
