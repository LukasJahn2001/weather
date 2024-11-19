
import matplotlib.pyplot as plt 
import csv 
import math

  
x = [] 
y = []
  
with open('/home/lukas/git/weather/losses_validation.csv','r') as csvfile: 
    lines = csv.reader(csvfile, delimiter=',') 
    for row in lines: 
        x.append(row[0]) 
        y.append(float(row[1])) 

a = []
b = []

# ceil((length - multistep)/ batch-size )* multistep)
numberOfAvg = 8 #31181

for count in range(int(math.floor(len(y)/numberOfAvg))):
    sum = 0
    for i in range(numberOfAvg):
        sum = sum +  y[count * numberOfAvg + i]
    
    b.append(sum/numberOfAvg)
    a.append(count)
    


  
plt.plot(a, b, color = 'g', linestyle = 'dashed', 
         marker = 'o',label = "Weather Data") 
  
plt.xticks(rotation = 90) 
plt.xlabel('Epoch') 
plt.ylabel('loss') 
plt.title('loss over time', fontsize = 20) 
plt.grid() 
plt.legend() 
plt.show() 