import matplotlib.pyplot as plt
import sys
import statistics as stats
import time
import pickle

file_name = "data.txt"

with open(file_name,'rb') as file:
   xs = pickle.load(file)
   ys = pickle.load(file)

# xs = [1,2,3,4]
# ys = [1,4,9,16]
learning_rates = [.002,.001,.001]
num_iterations=[140,150,200]
normalize_b = True
starting_C = [0,0,0,0]
colours = ['k','r','g','c']

all_coefs =[[]]*len(starting_C)
x_max = max(xs)
y_max = max(ys)
x_min = min(xs)
y_min = min(ys)
points = [xs,ys]
def normalize(arr):
   max1 = max(arr)
   min1 = min(arr)
   return [(x-min1)/(max1-min1) for x in arr]

def getLine(C):
   line_xs = []
   line_ys = []
   x = min(points[0])-1
   step_size = (max(points[0])+1 - min(points[0])-1)/100
   while (x<max(points[0])+1):
      line_xs.append(x)
      line_ys.append(sum(list(map(lambda pair: pair[0]*(x**pair[1]),zip(C,range(len(C)))))))
      x+=step_size
   return [line_xs,line_ys]

def plot(C):
   plt.grid(linestyle='dashed')
   axes = plt.gca()
   axes.set_xlim([x_min-.1,x_max+.1])
   axes.set_ylim([y_min-.1,y_max+.1])
   plt.plot(points[0],points[1],'bo')
   for i in range(len(all_coefs)):
      c = all_coefs[i]
      if c:
         line_x,line_y = getLine(c)
         plt.plot(line_x,line_y,colours[i])
   plt.pause(.00000001)
   plt.gcf().clear()


if normalize_b:
   xs = normalize(xs)
   ys = normalize(ys)
   points = [xs,ys]
   x_max = 1
   y_max = 1
   x_min = 0
   y_min = 0

# ys = [5,7,9,11,13,15]
# xs = normalize(xs)
# ys = normalize(ys)
# tmp = [x for x,y in sorted(zip(xs,ys))]
# ys = [y for x,y in sorted(zip(xs,ys))]
# xs = tmp




# def sumSquaredError(actual,a,b):
   # return 1/2*sum(list(map(lambda pair: (pair[0]-pair[1])**2,zip(actual[1],[x*a+b for x in actual[0]]))))

def sumSquaredError(actual,C):
   guesses = [0]*len(actual[0])
   for i in range(len(actual[0])):
      for j in range(len(C)):
         guesses[i] += C[j]*actual[0][i]**j
   # print("Guesses: " + str(guesses))
   return 1/2*sum(list(map(lambda pair: (pair[0]-pair[1])**2,zip(actual[1],guesses))))

print(["{0:0.2f}".format(i) for i in xs])
print(["{0:0.2f}".format(i) for i in ys])

def gradientStep(C,learning_rate, points):
   x_arr = points[0]
   y_arr = points[1]
   num_vars = len(C)
   gradients = [0]*num_vars
   for i in range(len(x_arr)):
      y_p = sum(list(map(lambda pair: pair[0]*(x_arr[i]**pair[1]),zip(C,range(num_vars)))))
      # print("i: " + str(i) + ", y_arr[i]: " + str(y_arr[i]) + ", y_p: " + str(y_p))
      for j in range(num_vars):
         gradients[j] -= (y_arr[i]-y_p)*x_arr[i]**j
         # print(str(j) + ") Gradients[j]: " + str(gradients[j]))
   for i in range(len(C)):
      C[i] = C[i] -learning_rate*gradients[i]
   return C
def linearRegression(points,starting_C,learning_rate, num_iterations):
   # for j in range(0,1):
   for j in reversed(range(len(starting_C)-1)):
      C = starting_C[j::]
      for i in range(num_iterations[j]):
         C = gradientStep(C,learning_rates[j],points)
         all_coefs[j] = C
         if not i%(int(num_iterations[j]/40)):
            plot(C)
            print (str(i) + ") stepped to: " + str(C) + " with error: " + str(sumSquaredError(points,C)))
   return C
   

# print(["{0:0.2f}".format(i) for i in xs])
# print(["{0:0.2f}".format(i) for i in ys])

# plt.plot(points[0],points[1],'ro')
# plt.show()


C = linearRegression(points,starting_C,learning_rates,num_iterations)

