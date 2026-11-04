

import pandas as pd

# Customer Transaction Data - Based on Purchasing Behavior
# 20 Customers = 4 Meaningful Groups x 5 Customers

data = {
    'CustomerID': [1,2,3,4,5, 6,7,8,9,10, 11,12,13,14,15, 16,17,18,19,20],
    'Age': [22,24,25,27,28, 32,35,38,40,42, 45,48,50,55,58, 23,25,27,29,30],
    'AnnualIncome': [25000,28000,22000,30000,27000, 50000,55000,60000,48000,52000, 80000,85000,90000,78000,82000, 65000,70000,68000,72000,62000],
    'TotalSpent': [9000,11000,8500,12000,10000, 28000,32000,35000,26000,30000, 60000,70000,75000,65000,68000, 42000,48000,45000,52000,40000],
    'NumTransactions': [30,32,28,35,31, 18,20,22,17,19, 10,12,9,11,13, 40,45,38,48,42],
    'AvgTransactionValue': [300,340,290,360,320, 1100,1200,1300,1050,1150, 4200,4500,4800,4000,4300, 950,1050,1000,1100,900],
    'RecencyDays': [5,7,4,6,8, 10,12,9,14,11, 20,25,18,22,28, 3,2,4,5,3]
}

df = pd.DataFrame(data)
df.to_csv('customer_transaction_data.csv', index=False)

print("CSV Created Successfully!")
print(df)
print(df.shape)  # (20, 7)