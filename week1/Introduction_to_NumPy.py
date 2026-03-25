# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python [conda env:opencv]
#     language: python
#     name: conda-env-opencv-py
# ---

# # <font color='blue'>Numpy Refresher</font>
#
# ### <font style="color:rgb(8,133,37)">Why do we need a special library for math and DL?</font>
# Python provides data types such as lists / tuples out of the box. Then, why are we using special libraries for deep learning tasks, such as Pytorch or TensorFlow, and not using standard types?
#
# The major reason is efficiency - In pure python, there are no primitive types for numbers, as in e.g. C language. All the data types in Python are objects with lots of properties and methods. You can see it using the `dir` function:

a = 3
dir(a)[-10:]

# ### <font style="color:rgb(8,133,37)">Python Issues</font>
#
# - slow in tasks that require tons of simple math operations on numbers
# - huge memory overhead due to storing plain numbers as objects
# - runtime overhead during memory dereferencing - cache issues
#
#
# NumPy is an abbreviation for "numerical python" and as it stands from the naming it provides a rich collection of operations on the numerical data types with a python interface. The core data structure of NumPy is `ndarray` - a multidimensional array. Let's take a look at its interface in comparison with plain python lists.

# # <font color='blue'>Performance comparison of Numpy array and Python lists </font>
#
# Let's imagine a simple task - we have several 2-dimensional points and we want to represent them as a list of points for further processing. For the sake of simplicity of processing we will not create a `Point` object and will use a list of 2 elements to represent coordinates of each point (`x` and `y`):

# create points list using explicit specification of coordinates of each point
points = [[0, 1], [10, 5], [7, 3]]
points

# +
# create random points
from random import randint

num_dims = 2
num_points = 10
x_range = (0, 10)
y_range = (1, 50)
points = [[randint(*x_range), randint(*y_range)] for _ in range(num_points)]
points
# -

# **How can we do the same using Numpy? Easy!**

import numpy as np
points = np.array(points)  # we are able to create numpy arrays from python lists
points

# create random points using numpy library
num_dims = 2
num_points = 10
x_range = (0, 11)
y_range = (1, 51)
points = np.random.randint(
    low=(x_range[0], y_range[0]), high=(x_range[1], y_range[1]), size=(num_points, num_dims)
)
points

# **It may look as over-complication to use NumPy for the creation of such a list and we still cannot see the good sides of this approach. But let's take a look at the performance side.**

num_dims = 2
num_points = 100000
x_range = (0, 10)
y_range = (1, 50)

# ### <font style="color:rgb(8,133,37)">Python performance</font>

# %timeit \
# points = [[randint(*x_range), randint(*y_range)] for _ in range(num_points)]

# ### <font style="color:rgb(8,133,37)">NumPy performance</font>

# %timeit \
# points = np.random.randint(low=(x_range[0], y_range[0]), high=(x_range[1], y_range[1]), size=(num_points, num_dims))

# Wow, NumPy is **around 50 times faster** than pure Python on this task! One may say that the size of the array we're generating is relatively large, but it's very reasonable if we take into account the dimensions of inputs (and weights) in neural networks (or math problems such as hydrodynamics).

# # <font style="color:blue">Basics of Numpy </font>
# We will go over some of the useful operations of Numpy arrays, which are most commonly used in ML tasks.

# ## <font color='blue'>1. Basic Operations </font>
#

# ### <font style="color:rgb(8,133,37)">1.1. Python list to numpy array</font>

# +
py_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

np_array = np.array(py_list)
np_array

# +
py_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]

np_array= np.array(py_list)
np_array
# -

# ### <font style="color:rgb(8,133,37)">1.2. Slicing and Indexing</font>

print('First row:\t\t\t{}'.format(np_array[0]))
print('First column:\t\t\t{}'.format(np_array[:, 0]))
print('3rd row 2nd column element:\t{}'.format(np_array[2][1]))
print('2nd onwards row and 2nd onwards column:\n{}'.format(np_array[1:, 1:]))
print('Last 2 rows and last 2 columns:\n{}'.format(np_array[-2:, -2:]))
print('Array with 3rd, 1st and 4th row:\n{}'.format(np_array[[2, 0, 3]]))

# ### <font style="color:rgb(8,133,37)">1.3. Basic attributes of NumPy array</font>
#
# Get a full list of attributes of an ndarray object [here](https://numpy.org/devdocs/user/quickstart.html).

print('Data type:\t{}'.format(np_array.dtype))
print('Array shape:\t{}'.format(np_array.shape))


# Let's create a function (with name `array_info`) to print the NumPy array, its shape, and its data type. We use this function to print arrays further in this section. 
#

# +
def array_info(array):
    print('Array:\n{}'.format(array))
    print('Data type:\t{}'.format(array.dtype))
    print('Array shape:\t{}\n'.format(array.shape))
    
array_info(np_array)
# -

# ### <font style="color:rgb(8,133,37)">1.4. Creating NumPy array using built-in functions and datatypes</font>
#
# The full list of supported data types can be found [here](https://numpy.org/devdocs/user/basics.types.html).
#

# **Sequence Array**
#
# `np.arange([start, ]stop, [step, ]dtype=None)`
#
# Return evenly spaced values in `[start, stop)`.
#
# More delatis of the function can be found [here](https://docs.scipy.org/doc/numpy/reference/generated/numpy.arange.html).

# sequence array
array = np.arange(10, dtype=np.int64)
array_info(array)

# sequence array
array = np.arange(5, 10, dtype=np.float32)
array_info(array)

# **Zeroes Array**

# Zero array/matrix
zeros = np.zeros((2, 3), dtype=np.float32)
array_info(zeros)

# **Ones Array**

# ones array/matrix
ones = np.ones((3, 2), dtype=np.int8)
array_info(ones)

# **Constant Array**

# constant array/matrix
array = np.full((3, 3), 3.14)
array_info(array)

# **Identity Array**

# identity array/matrix
identity = np.eye(5, dtype=np.float32)      # identity matrix of shape 5x5
array_info(identity)

# **Random Integers Array**
#
# `np.random.randint(low, high=None, size=None, dtype='l')`
#
# Return random integer from the `discrete uniform` distribution in `[low, high)`. If high is `None`, then return elements are in `[0, low)`
#
# More details can be found [here](https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.random.randint.html).

# random integers array/matrix
rand_int = np.random.randint(5, 10, (2,3)) # random integer array of shape 2x3, values lies in [5, 10)
array_info(rand_int)

# **Random Array**
#
# `np.random.random(size=None)`
#
# Results are from the `continuous uniform` distribution in `[0.0, 1.0)`.
#
# These types of functions are useful is initializing the weight in Deep Learning. More details and similar functions can found [here](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.random.html).

# random array/matrix
random_array = np.random.random((5, 5))   # random array of shape 5x5
array_info(random_array)

# **Boolean Array**
#
# If we compare above `random_array` with some `constant` or `array` of the same shape, we will get a boolean array.

# Boolean array/matrix
bool_array = random_array > 0.5
array_info(bool_array)

# The boolean array can be used to get value from the array. If we use a boolean array of the same shape as indices, we will get those values for which the boolean array is True, and other values will be masked.
#
# Let's use the above `boolen_array` to get values from `random_array`.

# Use boolean array/matrix to get values from array/matrix
values = random_array[bool_array]
array_info(values)

# Basically, from the above method, we are filtering values that are greater than `0.5`. 

# **Linespace**
#
# `np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)`
#
# Returns num evenly spaced samples, calculated over the interval `[start, stop]`.
#
# More detais about the function find [here](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html)

# Linspace
linespace = np.linspace(0, 5, 7, dtype=np.float32)   # 7 elements between 0 and 5
array_info(linespace)

# ### <font style="color:rgb(8,133,37)">1.5. Data type conversion</font>
#
# Sometimes it is essential to convert one data type to another data type.

age_in_years = np.random.randint(0, 100, 10)
array_info(age_in_years)

# Do we really need an `int64` data type to store age?
#
# So let's convert it to `uint8`.

age_in_years = age_in_years.astype(np.uint8)
array_info(age_in_years)

# Let's convert it to `float128`. 😜

age_in_years = age_in_years.astype(np.float128)
array_info(age_in_years)

# ## <font color='blue'>2. Mathematical functions </font>
#
# Numpy supports a lot of Mathematical operations with array/matrix. Here we will see a few of them which are useful in Deep Learning. All supported functions can be found [here](https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.math.html).

# ### <font style="color:rgb(8,133,37)">2.1. Exponential Function </font>
# Exponential functions ( also called `exp` ) are used in neural networks as activations functions. They are used in softmax functions which is widely used in Classification tasks.
#
# Return element-wise `exponential` of `array`.
#
# More details of `np.exp` can be found **[here](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.exp.html#numpy.exp)**

# +
array = np.array([np.full(3, -1), np.zeros(3), np.ones(3)])
array_info(array)

# exponential of a array/matrix
print('Exponential of an array:')
exp_array = np.exp(array)
array_info(exp_array)
# -

# ### <font style="color:rgb(8,133,37)">2.2. Square Root </font>
#
# `np.sqrt` return the element-wise `square-root` (`non-negative`) of an array.
#
# More details of the function can be found [here](https://docs.scipy.org/doc/numpy/reference/generated/numpy.sqrt.html)
#
# `Root Mean Square Error` (`RMSE`) and `Mean Absolute Error` (`MAE`) commonly used to measure the `accuracy` of continuous variables.

# +
array = np.arange(10)
array_info(array)

print('Square root:')
root_array = np.sqrt(array)
array_info(root_array)
# -

# ### <font style="color:rgb(8,133,37)">2.3. Logrithm </font>
#
# `np.log` return element-wise natural logrithm of an array.
#
# More details of the function can be found [here](https://docs.scipy.org/doc/numpy/reference/generated/numpy.log.html)
#
# `Cross-Entropy` / `log loss` is the most commonly used loss in Machine Learning classification problem. 

# +
array = np.array([0, np.e, np.e**2, 1, 10])
array_info(array)

print('Logrithm:')
log_array = np.log(array)
array_info(log_array)
# -

# <font color='red'>**Note:** Getting warning because we are trying to calculate `log(0)`.</font>

# ### <font style="color:rgb(8,133,37)">2.4. Power </font>
#
# `numpy.power(x1, x2)`
#
# Returns first array elements raised to powers from second array, element-wise.
#
# Second array must be broadcastable to first array.
#
# What is **broadcasting**? We will see later.
#
# More detalis about the function can be found [here](https://docs.scipy.org/doc/numpy/reference/generated/numpy.power.html)

# +
array = np.arange(0, 6, dtype=np.int64)
array_info(array)

print('Power 3:')
pow_array = np.power(array, 3)
array_info(pow_array)
# -

# ### <font style="color:rgb(8,133,37)">2.5. Clip Values </font>
#
# `np.clip(a, a_min, a_max)`
#
# Return element-wise cliped values between `a_min` and `a_max`.
#
# More details of the finction can be found [here](https://docs.scipy.org/doc/numpy/reference/generated/numpy.clip.html)
#
# `Rectified Linear Unit` (`ReLU`) is the most commonly used activation function in Deep Learning.
#
# What ReLU do?
#
# If the value is less than zero, it makes it zero otherwise leave as it is. In NumPy assignment will be implementing this activation function using NumPy.

# +
array = np.random.random((3, 3))
array_info(array)

# clipped between 0.2 and 0.5
print('Clipped between 0.2 and 0.5')
cliped_array = np.clip(array, 0.2, 0.5)
array_info(cliped_array)

# clipped to 0.2
print('Clipped to 0.2')
cliped_array = np.clip(array, 0.2, np.inf)
array_info(cliped_array)
# -

# ## <font color='blue'>3. Reshape ndarray </font>
#
# Reshaping the array / matrix is very often required in Machine Learning and Computer vision. 

# ### <font style="color:rgb(8,133,37)">3.1. Reshape </font>
#
# `np.reshape` gives an array in new shape, without changing its data.
#
# More details of the function can be found [here](https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html)

# +
a = np.arange(1, 10, dtype=np.int32)
array_info(a)

print('Reshape to 3x3:')
a_3x3 = a.reshape(3, 3)
array_info(a_3x3)

print('Reshape 3x3 to 3x3x1:')
a_3x3x1 = a_3x3.reshape(3, 3, 1)
array_info(a_3x3x1)
# -

# ### <font style="color:rgb(8,133,37)">3.2. Expand Dim </font>
#
# `np.expand_dims`
#
# In the last reshape, we have added a new axis. We can use `np.expand_dims` or `np.newaxis` to do the same thing.
#
# Mode details for `np.expand_dim` can be found [here](https://docs.scipy.org/doc/numpy/reference/generated/numpy.expand_dims.html)

# +
print('Using np.expand_dims:')
a_expand = np.expand_dims(a_3x3, axis=2)
array_info(a_expand)

print('Using np.newaxis:')
a_newaxis = a_3x3[..., np.newaxis]
# or 
# a_newaxis = a_3x3[:, :, np.newaxis]
array_info(a_newaxis)
# -

# ### <font style="color:rgb(8,133,37)">3.3. Squeeze </font>
#
# Sometimes we need to remove the redundant axis (single-dimensional entries). We can use `np.squeeze` to do this.
#
# More details of `np.squeeze` can be found [here](https://docs.scipy.org/doc/numpy/reference/generated/numpy.squeeze.html)
#
# Deep Learning very often uses this functionality.

# +
print('Squeeze along axis=2:')
a_squeezed = np.squeeze(a_newaxis, axis=2)
array_info(a_squeezed)

# should get value error
print('Squeeze along axis=1, should get ValueError')
a_squeezed_error = np.squeeze(a_newaxis, axis=1)  # Getting error because of the size of 
                                                  # axis-1 is not equal to one.
# -

# <font color='red'>**Note:** Getting error because of the size of axis-1 is not equal to one.</font>

# ### <font style="color:rgb(8,133,37)">3.4. Reshape revisit </font>
#
# We have a 1-d array of length n. We want to reshape in a 2-d array such that the number of columns becomes two, and we do not care about the number of rows. 

# +
a = np.arange(10)
array_info(a)

print('Reshape such that number of column is 2:')
a_col_2 = a.reshape(-1, 2)
array_info(a_col_2)
# -

# ## <font color='blue'>4. Combine Arrays / Matrix </font>
#
# Combining two or more arrays is a frequent operation in machine learning. Let's have a look at a few methods. 
#

# ### <font style="color:rgb(8,133,37)">4.1. Concatenate </font>
#
# `np.concatenate`, Join a sequence of arrays along an existing axis.
#
# More details of the function find [here](https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html)

# +
a1 = np.array([[1, 2, 3], [4, 5, 6]])
a2 = np.array([[7, 8, 9]])

print('Concatenate along axis zero:')
array = np.concatenate((a1, a2), axis=0)
array_info(a1)
array_info(a2)
array_info(array)
# -

# ### <font style="color:rgb(8,133,37)">4.2. hstack </font>
#
# `np.hstack`, stack arrays in sequence horizontally (column-wise).
#
# More details of the function find [here](https://docs.scipy.org/doc/numpy/reference/generated/numpy.hstack.html#numpy.hstack)

# +
a1 = np.array((1, 2, 3))
a2 = np.array((4, 5, 6))
a_hstacked = np.hstack((a1,a2))

print('Horizontal stack:')
array_info(a_hstacked)

# +
a1 = np.array([[1],[2],[3]])
a2 = np.array([[4],[5],[6]])
a_hstacked = np.hstack((a1,a2))

print('Horizontal stack:')
array_info(a_hstacked)
# -

# ### <font style="color:rgb(8,133,37)">4.3. vstack </font>
#
# `np.vstack`, tack arrays in sequence vertically (row-wise).
#
# More details of the function find [here](https://docs.scipy.org/doc/numpy/reference/generated/numpy.vstack.html#numpy.vstack)

# +
a1 = np.array([1, 2, 3])
a2 = np.array([4, 5, 6])
a_vstacked = np.vstack((a1, a2))

print('Vertical stack:')
array_info(a_vstacked)

# +
a1 = np.array([[1, 11], [2, 22], [3, 33]])
a2 = np.array([[4, 44], [5, 55], [6, 66]])
a_vstacked = np.vstack((a1, a2))

print('Vertical stack:')
array_info(a1)
array_info(a2)
array_info(a_vstacked)
# -

# ## <font color='blue'>5. Element wise Operations </font>
#

# Let's generate a random number to show element-wise operations. 

a = np.random.random((4,4))
b = np.random.random((4,4))
array_info(a)
array_info(b)

# ### <font style="color:rgb(8,133,37)">5.1. Element wise Scalar Operation </font>

# **Scalar Addition**

a + 5 # element wise scalar addition

# **Scalar Subtraction**

a - 5 # element wise scalar subtraction

# **Scalar Multiplication**

a * 10 # element wise scalar multiplication

# **Scalar Division**

a/10 # element wise scalar division

# ### <font style="color:rgb(8,133,37)">5.2. Element wise Array Operations </font>

# **Arrays Addition**

a + b # element wise array/vector addition

# **Arrays Subtraction**

a - b # element wise array/vector subtraction

# **Arrays Multiplication**

a * b # element wise array/vector multiplication

# **Arrays Division**

a / b # element wise array/vector division

# We can notice that the dimension of both arrays is equal in above arrays element-wise operations. **What if dimensions are not equal.** Let's check!!

print('Array "a":')
array_info(a)
print('Array "c":')
c = np.random.rand(2, 2)
array_info(c)
# Should throw ValueError
a + c

# <font color='red'>**Oh got the ValueError!!**</font>
#
# What is this error?
#
# <font color='red'>ValueError</font>: operands could not be broadcast together with shapes `(4,4)` `(2,2)` 
#
# **Let's see it next.**
#

# ### <font style="color:rgb(8,133,37)">5.3. Broadcasting </font>
#
# There is a concept of broadcasting in NumPy, which tries to copy rows or columns in the lower-dimensional array to make an equal dimensional array of higher-dimensional array. 
#
# Let's try to understand with a simple example.

# +
a = np.array([[1, 2, 3], [4, 5, 6],[7, 8, 9]])
b = np.array([0, 1, 0])

print('Array "a":')
array_info(a)
print('Array "b":')
array_info(b)

print('Array "a+b":')
array_info(a+b)  # b is reshaped such that it can be added to a.


# b = [0,1,0] is broadcasted to     [[0, 1, 0],
#                                    [0, 1, 0],
#                                    [0, 1, 0]]  and added to a.
# -

# ## <font color='blue'>6. Linear Algebra</font>
#
# Here we see commonly use linear algebra operations in Machine Learning. 

# ### <font style="color:rgb(8,133,37)">6.1. Transpose </font>

# +
a = np.random.random((2,3))
print('Array "a":')
array_info(a)

print('Transose of "a":')
a_transpose = a.transpose()
array_info(a_transpose)
# -

# ### <font style="color:rgb(8,133,37)">6.2. Matrix Multiplication</font>
# We will discuss 2 ways of performing Matrix Multiplication.
#
# - `matmul`
# - Python `@` operator
#
# **Using matmul function in numpy**
# This is the most used approach for multiplying two matrices using Numpy. [See docs](https://docs.scipy.org/doc/numpy/reference/generated/numpy.matmul.html)

# +
a = np.random.random((3, 4))
b = np.random.random((4, 2))

print('Array "a":')
array_info(a)
print('Array "b"')
array_info(b)

c = np.matmul(a,b) # matrix multiplication of a and b

print('matrix multiplication of a and b:')
array_info(c)

print('{} x {} --> {}'.format(a.shape, b.shape, c.shape)) # dim1 of a and dim0 of b has to be 
                                                        # same for matrix multiplication
# -

# **Using `@` operator**
# This method of multiplication was introduced in Python 3.5. [See docs](https://www.python.org/dev/peps/pep-0465/)

# +
a = np.random.random((3, 4))
b = np.random.random((4, 2))

print('Array "a":')
array_info(a)
print('Array "b"')
array_info(b)

c = a@b # matrix multiplication of a and b
array_info(c)
# -

# ### <font style="color:rgb(8,133,37)">6.3. Inverse</font>

# +
A = np.random.random((3,3))
print('Array "A":')
array_info(A)
A_inverse = np.linalg.inv(A)
print('Inverse of "A" ("A_inverse"):')
array_info(A_inverse)

print('"A x A_inverse = Identity" should be true:')
A_X_A_inverse = np.matmul(A, A_inverse)  # A x A_inverse = I = Identity matrix
array_info(A_X_A_inverse)
# -

# ### <font style="color:rgb(8,133,37)">6.4. Dot Product</font>

# +
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

dot_pro = np.dot(a, b)  # It will be a scalar, so its shape will be empty
array_info(dot_pro)
# -

# ## <font color='blue'>7. Array statistics</font>

# ### <font style="color:rgb(8,133,37)">7.1. Sum</font>

# +
a = np.array([1, 2, 3, 4, 5])

print(a.sum())
# -

# ### <font style="color:rgb(8,133,37)">7.2. Sum along Axis</font>

# +
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a)
print('')

print('sum along 0th axis = ',a.sum(axis = 0)) # sum along 0th axis ie: 1+4, 2+5, 3+6
print("")
print('sum along 1st axis = ',a.sum(axis = 1)) # sum along 1st axis ie: 1+2+3, 4+5+6
# -

# ### <font style="color:rgb(8,133,37)">7.3. Minimum and Maximum</font>

# +
a = np.array([-1.1, 2, 5, 100])

print('Minimum = ', a.min())
print('Maximum = ', a.max())
# -

# ### <font style="color:rgb(8,133,37)">7.4. Min and Max along Axis</font>

# +
a = np.array([[-2, 0, 2], [1, 2, 3]])

print('a =\n',a,'\n')
print('Minimum = ', a.min())
print('Maximum = ', a.max())
print()
print('Minimum along axis 0 = ', a.min(0))
print('Maximum along axis 0 = ', a.max(0))
print()
print('Minimum along axis 1 = ', a.min(1))
print('Maximum along axis 1 = ', a.max(1))
# -

# ### <font style="color:rgb(8,133,37)">7.5. Mean and Standard Deviation</font>

# +
a = np.array([-1, 0, -0.4, 1.2, 1.43, -1.9, 0.66])

print('mean of the array = ',a.mean())
print('standard deviation of the array = ',a.std())
# -

# ### <font style="color:rgb(8,133,37)">7.6. Standardizing the Array</font>
#
# Make distribution of array elements such that`mean=0` and `std=1`.

# +
a = np.array([-1, 0, -0.4, 1.2, 1.43, -1.9, 0.66])

print('mean of the array = ',a.mean())
print('standard deviation of the array = ',a.std())
print()

standardized_a = (a - a.mean())/a.std()
print('Standardized Array = ', standardized_a)
print()

print('mean of the standardized array = ',standardized_a.mean()) # close to 0
print('standard deviation of the standardized  array = ',standardized_a.std()) # equals to 1
# -

# # <font color='blue'>References </font>
#
# https://numpy.org/devdocs/user/quickstart.html
#
# https://numpy.org/devdocs/user/basics.types.html
#
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.astype.html
#
# https://coolsymbol.com/emojis/emoji-for-copy-and-paste.html
#
# https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.math.html
#
# https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.exp.html#numpy.exp
#
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.clip.html
#
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.sqrt.html
#
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.log.html
#
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.power.html
#
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html
#
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.expand_dims.html
#
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.squeeze.html
#
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html
#
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.hstack.html#numpy.hstack
#
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.vstack.html#numpy.vstack


