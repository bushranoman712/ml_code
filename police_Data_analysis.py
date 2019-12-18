# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 12:19:08 2019

Dear Reader,

When I wrote this code, only I and God knew what it was.

Now, only God knows!

total hours worked consistently =  8   

@author: Noman
"""

import pandas as pd
import numpy as np
import datetime
from dateutil.parser import parse
from math import radians, cos, sin, asin, sqrt






columns = ["Report_ID","Date","Time","Area_ID","Area_Name",
           "Reporting_District","Age","Sex_Code","Descent_Code","Charge_Group_Code"
           ,"Charge_Group_Description","Arrest_Type_Code","Charge", "Charge_Description",
           "Address","Cross_Street","Location"]

df = pd.read_csv("F:\challenge\\Arrest_Data_from_2010_to_Present.csv", sep = ',')

#rename columns
df.columns = columns



#print(df.head())

df['Arrest_Date'] = pd.to_datetime(df['Date']).dt.date



#question no 1 == How many bookings of arrestees were made in 2018?


#start_date = datetime.datetime.strptime('01-01-2018', "%m-%d-%Y").date()
#end_date = datetime.datetime.strptime('12-31-2018', "%m-%d-%Y").date()
#print(start_date)
#print(end_date)

start_date = pd.to_datetime("31-12-2017").date()
end_date = pd.to_datetime("01-01-2019").date()
q1df = df[(df['Arrest_Date'] > start_date) & (df['Arrest_Date'] < end_date)] 
#print(q1df.head()[["Arrest_Date"]])
#print(q1df.tail()[["Arrest_Date"]])
#print(q1df.shape) #number of rows represent total arrests in 2018
row,col = q1df.shape

print("******************# Question 1 #*********************")
print("\n")
print("Answer:" + str(row))
print("\n")
print("*****************************************************")




#question no 2 == How many bookings of arrestees were made in the area with the most arrests in 2018?
print("******************# Question 2 #*********************")
print("\n")
print(q1df.groupby("Area_Name").count().sort_values("Age" , ascending = False)[["Age"]])
#Central Area (10951) arrests in 2018...
answer = q1df.groupby("Area_Name").count().sort_values("Age" , ascending = False)[["Age"]]
zlist = []
for index,row in answer.iterrows():
    zlist.append(row['Age'])


print("\n")
print("Answer:" + str(zlist[0]))
print("\n")
print("*****************************************************")





#question no 3 == What is the 95% quantile of the age of the arrestee in 2018? Only consider the following charge groups for your analysis:
#q1df = q1df[pd.notnull(q1df['Charge_Group_Description'])]
#print(q1df.head())
#print(q1df.tail())


q3df = q1df[ (q1df.Charge_Group_Description == "Vehicle Theft") | (q1df.Charge_Group_Description == "Robbery") |
            (q1df.Charge_Group_Description == "Burglary") | (q1df.Charge_Group_Description == "Receive Stolen Property") ] 
#print(q3df.head())
#print(q3df.tail())
#print(q3df.shape)
pd.set_option('display.max_rows', 75)
pd.set_option("display.precision", 10)
#q3df  = q3df.groupby("Age").count()
#print(q3df[["Charge_Group_Description"]])
#print(q3df.groupby("Age").count()[["Arrest_Date"]])
#with pd.option_context('display.float_format', '{:0.10f}'.format):
#    print(q3df.Age.quantile(0.95))

print("******************# Question 3 #*********************")
print("\n")
print("Quantile:" + str(q3df.Age.quantile(0.95)))
print("\n")
print("*****************************************************")






#question no 4 ==For this question, calculate the Z-score 
#of the average age for each charge group. Report the largest absolute value among the calculated Z-scores.
print("******************# Question 4 #*********************")
q1df = q1df[pd.notnull(q1df['Charge_Group_Description'])]
#print(q1df.shape)
q4df = q1df[(q1df.Charge_Group_Description != "Non-Criminal Detention")]
q4df = q4df[(q4df.Charge_Group_Description != "Pre-Delinquency")] 
#print(q4df.shape)
print(q4df.groupby(["Charge_Group_Description"])[["Age"]].mean())


q4df = q4df.groupby(["Charge_Group_Description"])[["Age"]].mean()

zlist = []
for index,row in q4df.iterrows():
    zlist.append(row['Age'])

from scipy import stats
print ("\nZ-score for average ages for each group: \n", stats.zscore(zlist, axis = 0))
l_o_l = stats.zscore(zlist, axis = 0)
arr = np.array(l_o_l)
xmax = arr.flat[abs(arr).argmax()]



print("\n")
print("largest Z-Score:" + str(xmax))
print("\n")
print("*****************************************************")







#question no 5 
print("******************# Question 5 Answer #*********************")
q5df = df
q5df['year'] = pd.to_datetime(df['Arrest_Date']).dt.year
q5df = q5df[pd.notnull(q5df['Charge_Group_Description'])]


q5df = q5df[(q5df.Charge_Group_Description != "Non-Criminal Detention")]
q5df = q5df[(q5df.Charge_Group_Description != "Pre-Delinquency")]

start_year = pd.to_datetime("01-12-2009").date()
end_year = pd.to_datetime("01-01-2019").date()
q5df = q5df[(q5df['Arrest_Date'] > start_year) & (q5df['Arrest_Date'] < end_year)] 


print(q5df.groupby(["year"])[["Charge_Group_Description"]].count())

q5df = q5df.groupby(["year"])[["Charge_Group_Description"]].count()

yflist = []
for index,row in q5df.iterrows():
    yflist.append(row['Charge_Group_Description'])





Y = np.array(yflist)
print("Training data:")
print(Y)

X = np.array(list(map(int,["2010","2011","2012","2013","2014","2015","2016","2017","2018"])))
X = X.reshape((-1, 1))
print("labels:")
print(X)

from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(X, Y)
print('coefficient of determination:', model.score(X, Y))
print('intercept:', model.intercept_)
print('slope:', model.coef_)

X_Test = np.array(2019).reshape((-1, 1))
y_pred = model.predict(X_Test)


print("\n")
print('predicted response:', round(y_pred[0]), sep='\n')
print("\n")
print("*****************************************************")






def Calculate_KM(lon1, lat1, lon2, lat2):
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    d_lon = lon2 - lon1 
    d_lat = lat2 - lat1 
    cal = sin(d_lat/2)**2 + cos(lat1) * cos(lat2) * sin(d_lon/2)**2
    result = 2 * asin(sqrt(cal)) 
    # Radius of earth in kilometers is 6371
    earth_radius = 6371
    km = earth_radius * result
    return km




#question no 6

q6df = q1df
#print(q6df.head()[["Location"]])
#print(q6df.shape)
q6df = q6df[(q6df.Location != '(0.0, 0.0)')]
#print(q6df.shape)

latlon = []
for index,row in q6df.iterrows():
    latlon.append(row['Location'])

#print(len(latlon))
#print(latlon[0])

size  = int(len(latlon))
distance = []

for i in range(0, size):
    tp= latlon[i]
    values = ''.join(tp)
    values = values.replace("(","")
    values = values.replace(")","")
    #print(values)
    lat , lon = values.split(',')
    dis = Calculate_KM(34.050536,-118.247861,float(lat),float(lon))
    
    dis = distance.append(Calculate_KM(34.050536,-118.247861,float(lat),float(lon)))
    #print(dis)

#print(len(distance))


distances = np.array(distance)
q6df["distances"] = distances

df_for_question_7 = q6df


q6df = q6df[(q6df['distances'] > 0.0) & (q6df['distances'] < 2.0)] 
#print(q6df.shape)
row,col = q6df.shape
print("******************# Question 6 Answer #*********************")
print("\n")
print('Arrest incidents occurred within 2 km from the Bradbury Building in 2018: '+ str(row))
print("\n")
print("*****************************************************")






#question no 7


q7df = q1df
#print(q7df.shape)

q7df = q7df[(q7df.Location != '(0.0, 0.0)')]
#print(q7df.shape)

q7df = q7df[(q7df["Address"].str.contains('PICO'))]
#print(q7df.shape)


latlon = []
for index,row in q7df.iterrows():
    latlon.append(row['Location'])


size  = int(len(latlon))
latitudes = []
longitudes = []
for i in range(0, size):
    tp= latlon[i]
    values = ''.join(tp)
    values = values.replace("(","")
    values = values.replace(")","")
    #print(values)
    lat , lon = values.split(',')
    latitudes.append(lat)
    longitudes.append(lon)
    
    
q7df["latitudes"] = latitudes 
q7df["longitudes"] = longitudes 


lati_list = q7df["latitudes"].tolist()
lati_arr = np.array(lati_list)
lati_arr = lati_arr.astype(float)
lat_mean = lati_arr.mean()
#print(lat_mean)


lon_list = q7df["longitudes"].tolist()
lon_arr = np.array(lon_list)
lon_arr = lon_arr.astype(float)
lon_mean = lon_arr.mean()
#print(lon_mean)


q7df["lat_mean"] = lat_mean 
q7df["lon_mean"] = lon_mean 


q7df = q7df[(q7df['latitudes'] < str(lat_mean)) & (q7df['longitudes'] < str(lon_mean))]
#print(q7df.shape)
#print(q7df.head(10))


#calculate western adn eastern points
lat_wes = q7df.latitudes.max()
#print(lat_wes)


lon_est = q7df.longitudes.max()
#print(lon_est)




latlon = []
for index,row in q7df.iterrows():
    latlon.append(row['Location'])


size  = int(len(latlon))
distance = []

for i in range(0, size):
    tp= latlon[i]
    values = ''.join(tp)
    values = values.replace("(","")
    values = values.replace(")","")
    #print(values)
    lat , lon = values.split(',')
    #dis = haversine(lat_wes,lon_est,float(lat),float(lon))
    
    distance.append(round(Calculate_KM(float(lat_wes),float(lon_est),float(lat),float(lon))))


distances = np.array(distance)
q7df["distances_PICO"] = distances

#print(q7df.head())

print(q7df.groupby(["distances_PICO"])[["Address"]].count())
q7df = q7df.groupby(["distances_PICO"])[["Address"]].count()

crimes_list = []
for index,row in q7df.iterrows():
    crimes_list.append(row['Address'])

crimes =  np.array(crimes_list)
distance_per_km = np.array([0,1,2,3,4,5,7,12,15,16,19,20,21,22,23,26,27])


total_crimes = crimes.sum()/distance_per_km.sum()
#print(total_crimes)


print("******************# Question 7 Answer #*********************")
print("\n")
print('Arrest incidents per kilometer on Pico Boulevard in 2018.: '+ str(total_crimes))
print("\n")
print("*****************************************************")

