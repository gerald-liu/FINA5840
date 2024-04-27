#numbers
a = 1 #integer
b = 1.23 #float
c = complex(5,1)
d = 2.0
e = 7.0
a/b

#string ''
s1 = 'abcdefg'
s2 = 'hijklmn'
s3 = s1 + s2 #concatenation
s1.upper().lower()
s3 = s3.replace('b', 'bbb')

#list []
l1 = [1, '2', 'abc', [1,2,3]]
l2 = [3, 4, 5]
i = range(10) #closed on the left but open on the right
l3 = (3, 4, 5) #tuple

#dictionary {}
dic ={'a':1, 'b':2, 'c':3}
print(dic.items())
print(dic.keys())
print(dic.values())

#Flow control
a = 6
if a==5:
    print(a)
elif a==3:
    print(a+1)
elif a==4:
    print(a**2)
else:
    print('hello')

#for statement
for i in range(10):
    print(i**2)

for e in l1:
    print(e)

#while statement
k = 0
while k < 10:
    k += 0.333 #k = k+1
    print(k**2)
    if k > 5:
        continue

#try / except
sum = 0.0

for k in range(10):
    try:
        a = 1 - 1/(k-5)
        sum += a
    except:
        pass
        print(k)

#function
def sum_up(x, y, z):
    return x*z + y

sum_up2 = lambda x, y, z: x*z + y

def fib(n):
    fib = [1,1]
    for i in range(n):
        a = fib[-2] + fib[-1]
        fib.append(a)
    return fib

import time
#function with variable arguments
def myFun(*args): #*args: non-keyworded argument
    start = time.time()
    prod = 1.0
    for arg in args:
        prod *= arg
    cal_time = (time.time() - start)
    return {'res':prod, 'cal_time':cal_time}

myFun(1,2,3,4)

def myFun1(**kwargs): #kargs: key-worded arguments
    for key, value in kwargs.items():
        print("%s == %s" % (key, value))

myFun1(a=1, b=2, c=3)