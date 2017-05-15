import math
import pandas
A=[]
for x in range (1,10000,1) :
    A= A + [ x ]
for x in range ( 1, 5000, 1 ) :
    if ( ( x / 1000) >= 1 ) :
        b = int( x / 1000 )+(int(x/100) %10) +(int(x/10)%10)  + (x%10) + x
        A[ b - 1 ] = 0
    elif ((x/100)>=1):
        b= int(x/100)+(int(x/10)%10)+ x%10 + x
        A[b-1]=0
    elif ((x/10)>=1):
        b=int(x/10)+ x%10 + x
        A[b-1]=0
    else:
        b= x + x
        A[b - 1] = 0
print(sum(A[:5001]))







