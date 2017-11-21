import matplotlib.pyplot as plt
import sys
import statistics as stats
import time

def normalize(arr):
   max1 = max(arr)
   min1 = min(arr)
   return [(x-min1)/(max1-min1) for x in arr]

# xs = [1400,1600,1700,1875,1100,1550,2350,2450,1425,1700]
# ys = [245000,312000,279000,308000,199000,219000,405000,324000,319000,255000]
xs = [1,2,3,4,5]
ys = [5,16,33,56,85]

xs = normalize(xs)
ys = normalize(ys)
# tmp = [x for x,y in sorted(zip(xs,ys))]
# ys = [y for x,y in sorted(zip(xs,ys))]
# xs = tmp

points = [xs,ys]
starting_a = 10
starting_b = 0
learning_rate=.1
num_iterations=1




def sumSquaredError(actual,a,b):
   return 1/2*sum(list(map(lambda pair: (pair[0]-pair[1])**2,zip(actual[1],[x*a+b for x in actual[0]]))))

def gradientStep(a,b,learning_rate, points):
   x_arr = points[0]
   y_arr = points[1]
   a_gradient = 0
   b_gradient = 0
   for i in range(len(x_arr)):
      y_p = a*x_arr[i] + b
      a_gradient -= (y_arr[i] - y_p)*x_arr[i]
      b_gradient -= y_arr[i] - y_p
   a = a-learning_rate*a_gradient
   b = b-learning_rate*b_gradient
   return [a,b]

def linearRegression(points,starting_a,starting_b,learning_rate, num_iterations):
   a=starting_a
   b=starting_b
   for i in range(num_iterations):
      a,b = gradientStep(a,b,learning_rate,points)
      #####################################PLOTTING################################################
      if not i%25:
         plt.grid(linestyle='dashed')
         axes = plt.gca()
         axes.set_xlim([0,1])
         axes.set_ylim([0,1])
         plt.plot(points[0],points[1],'bo')
         x1,y1 = [0,1],[b,a+b]
         plt.plot(x1,y1,'r')
         plt.pause(.001)
         plt.gcf().clear()
      ############################################################################################
      print (str(i) + ") stepped to: y= " + str(a) + "x + " + str(b) + " with error: " + str(sumSquaredError(points,a,b)))
   return [a,b]
   

print(["{0:0.2f}".format(i) for i in xs])
print(["{0:0.2f}".format(i) for i in ys])





[a,b] = linearRegression(points,starting_a,starting_b,learning_rate,num_iterations)
