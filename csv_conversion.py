"""The purpose of this file is to convert textfile datasheets
 to csv files. These have greater data manipulation methods in
 Python."""

import os
import csv
#Set working directory (depends on system the code is being ran on)
os.chdir(r"C:\Users\35385\Documents\GitHub\Decision Trees")

#Open the text file and read the lines
with open('wildfires.txt', 'r') as textfile:
    lines = textfile.readlines()

#Create a list containing the data in the text file
data1=[]
for line in lines:
    data1.append(tuple(line.split()))

#Convert this data to a csv file format
#Specify the column names of the csv
#header = ['fire','year','temp','humidity','rainfall','drought_code','buildup_index','day','month','wind_speed']

#Open the new csv file in write mode and write the contents of the list to the csv
with open('wildfires.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    # write the header
    #writer.writerow()

    # write the data
    Count=0
    for line in data1:
        if Count==0:
            writer.writerow(line)
            Count+=1
        else:
            writer.writerow(line)
