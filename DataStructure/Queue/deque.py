#import module for print function
from __future__ import print_function

#import deque from collections
from collections import deque

dq = deque(list(range(5)))
print ("Initializing deque with range of 5")
print (dq)

dq.extendleft([-1,-2])
print ("Extending deque to left with [-2,-1]")
print (dq)

dq.rotate(-2)
print ("Rotating deque to right with 2 elements")
print (dq)

dq.rotate(2)
print ("Rotating deque to left with 2 elements")
print (dq)

dq.popleft()
print ("Removing element from left")
print (dq)

dq = deque(dq, maxlen = 5)
print ("Initializing the max len for dq")
print (dq)
