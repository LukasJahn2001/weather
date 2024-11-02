
import matplotlib.pyplot as plt 
import csv 
  
x = [] 
y = []
  
with open('/home/lukas/git/weather/losses.csv','r') as csvfile: 
    lines = csv.reader(csvfile, delimiter=',') 
    for row in lines: 
        x.append(row[0]) 
        y.append(float(row[1])) 


  
plt.plot(x, y, color = 'g', linestyle = 'dashed', 
         marker = 'o',label = "Weather Data") 
  
plt.xticks(rotation = 90) 
plt.xlabel('Dates') 
plt.ylabel('loss') 
plt.title('loss over time', fontsize = 20) 
plt.grid() 
plt.legend() 
plt.show() 