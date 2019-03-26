#create a list using list comprehension
numbers = [num for num in range(10)]
print(numbers)

#create a even number using list comprehension
even_num = [num for num in numbers if num%2 == 0 ]
print(even_num)

#create a nested for loop in list comprehension
nested = [(i, j) for i in [1,2,3] for j in [4,5,6]]
print(nested)

#create nested list comprehension 
import numpy as np
arr = np.array(range(1,10))
matrix = arr.reshape((3,3))
transpose = [[row[i] for row in matrix] for i in range(3)]
print(transpose)
