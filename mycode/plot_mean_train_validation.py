
import matplotlib.pyplot as plt 
import csv 
import math

  
x_t = [] 
y_t = []
x_v = [] 
y_v = []
  
with open('/home/lukas/git/weather/losses_train.csv','r') as csvfile: 
    lines = csv.reader(csvfile, delimiter=',') 
    for row in lines: 
        x_t.append(row[0]) 
        y_t.append(float(row[1])) 

with open('/home/lukas/git/weather/losses_validation.csv','r') as csvfile: 
    lines = csv.reader(csvfile, delimiter=',') 
    for row in lines: 
        x_v.append(row[0]) 
        y_v.append(float(row[1])) 

a = []
b = []
c = []

# ceil((length - multistep)/ batch-size )* multistep)
numberOfAvgTrain = 9314 #31181
numberOfAvgValidation = 183

print(type(y_t))

for count in range(int(math.floor(len(y_t)/numberOfAvgTrain))):
    sum = 0
    for i in range(numberOfAvgTrain):
        sum = sum +  y_t[count * numberOfAvgTrain + i]
    
    b.append(sum/numberOfAvgTrain)
    a.append(count)

for count in range(int(math.floor(len(y_v)/numberOfAvgValidation))):
    sum = 0
    for i in range(numberOfAvgValidation):
        sum = sum +  y_v[count * numberOfAvgValidation + i]
    c.append(sum/numberOfAvgValidation)
    


  
plt.plot(a, b, color = 'g', linestyle = 'dashed', 
         marker = 'o',label = "Training loss") 

plt.plot(a, c, color = 'r', linestyle = 'dashed', 
         marker = 'o',label = "Validation loss") 
  
plt.xticks(rotation = 90) 
plt.xlabel('Epoch') 
plt.ylabel('loss') 
plt.title('loss over time', fontsize = 20) 
plt.grid() 
plt.legend() 
plt.show() 