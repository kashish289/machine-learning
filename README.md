Load dataset
data = pd.read_csv('weather_data.csv')
 
data['date'] = pd.to_datetime(data['date'])

data.sort_values('date', inplace=True)
 
data.fillna(method='ffill', inplace=True)
 
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
 
data.fillna(method='ffill', inplace=True)

data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
 
X = data[['year', 'month', 'day']]
y = data['ppt']
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
model.fit(X_train, y_train)
 
y_pred = model.predict(X_test)
 
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')
 
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')
 
plt.figure(figsize=(10, 5))
