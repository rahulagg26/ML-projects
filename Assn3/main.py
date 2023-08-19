import time
import Adafruit_CharLCD as LCD

# Raspberry Pi pin configuration:
lcd_rs        = 12  
lcd_en        = 7
lcd_d4        = 8
lcd_d5        = 25
lcd_d6        = 17
lcd_d7        = 23
lcd_backlight = 4

lcd_columns = 20
lcd_rows    = 4

# Initialize the LCD using the pins above.
lcd = LCD.Adafruit_CharLCD(lcd_rs, lcd_en, lcd_d4, lcd_d5, lcd_d6, lcd_d7,
                           lcd_columns, lcd_rows, lcd_backlight)

lcd.message('Hello\nworld!')

time.sleep(5.0)

# Demo showing the cursor.
lcd.clear()
lcd.show_cursor(True)
lcd.message('Show cursor')

time.sleep(5.0)

# Demo showing the blinking cursor.
lcd.clear()
lcd.blink(True)
lcd.message('Blink cursor')

time.sleep(5.0)

# Stop blinking and showing cursor.
lcd.show_cursor(False)
lcd.blink(False)

# Demo scrolling message right/left.
lcd.clear()
message = 'Scroll'
lcd.message(message)
for i in range(lcd_columns-len(message)):
    time.sleep(0.5)
    lcd.move_right()
for i in range(lcd_columns-len(message)):
    time.sleep(0.5)
    lcd.move_left()

# Demo turning backlight off and on.
lcd.clear()
lcd.message('Flash backlight\nin 5 seconds...')
time.sleep(5.0)
# Turn backlight off.
lcd.set_backlight(0)
time.sleep(2.0)
# Change message.
lcd.clear()
lcd.message('Goodbye!')
# Turn backlight on.
lcd.set_backlight(1)

# The below code is implemented to train and to find the accuracy of the model:

import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.svm import LinearSVC

from pandas import read_csv
import pandas as pd
data = read_csv( "IOT_Assignment_2_data_regression_sensor_range.csv" )
x=data.loc[:,"Humidity(%)"]
y=data.loc[:,"Temperature(°C)"]
z=data.loc[:,"WaterFlow(%)"]
x = data[["Humidity(%)", "Temperature(°C)", "WaterFlow(%)"]]
X = x[["Humidity(%)", "Temperature(°C)"]]
Y = x["WaterFlow(%)"]
X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.3,train_size=0.7, random_state = 0)

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

model = linear_model.LinearRegression()

model.fit(X_train, y_train)
# predict
y_pred = model.predict(X_test)

# model evaluation
score = r2_score(y_test, y_pred)
print(score)
