from random import uniform
import pickle

num_items = 200
noise_percent = 50
noise_random = 20
x_min = -10
x_max = 10
file_name = "data.txt"
#Format c0 + c1X + c2X^2 + ...
C = [2,2,.5,.4]
xs = []
ys = []
for i in range(num_items):
   x = uniform(x_min,x_max)
   y = sum(list(map(lambda pair: pair[0]*(x**pair[1]),zip(C,range(len(C))))))
   y += uniform(-noise_random,noise_random)
   y *= uniform(1-noise_percent/100,1+noise_percent/100)
   xs.append(x)
   ys.append(y)

print("\n")
print(["{0:0.2f}".format(i) for i in xs])
print("\n")
print(["{0:0.2f}".format(i) for i in ys])
print("\n")

with open(file_name,'wb') as file:
   pickle.dump(xs,file)
   pickle.dump(ys,file)
