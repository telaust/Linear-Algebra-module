import math
import time


# 3Dimensional unit vectors
i = [1, 0, 0]
j = [0, 1, 0]
k = [0, 0, 1]


# Sum of 2 vectors
def vector_add(v, w):
	return [v_i + w_i for v_i , w_i in zip(v, w)]


# Subtraction of 2 vectors
def vector_subtract(v, w):
	return [v_i - w_i for v_i , w_i in zip(v, w)]


# Sum of set of vectors
def vector_sum(vectors):
	result = vectors[0]
	for vector in vectors[1:] :
		result = vector_add(result, vector)
	return result


# Multiplication of vector and scalar value
def scalar_mul(a, v):
	return [a*v_i for v_i in v]


def vector_normalize(v):
	vector_norm = norm(v)
	return [v_i/vector_norm for v_i in v]


# Utility function
def pack_vectors(*args): 
	l = []
	for i in args:
		l.append(i)
	return l


# Scalar product of 2 vectors
def dot_product(v, w):
	return sum( v_i * w_i for v_i, w_i in zip(v, w))


# Another way of dot product using angle, phi is degrees
def dot(v, w, phi):
	phi_degree = math.radians(phi)
	return norm(v) * norm(w) * round(math.cos(phi_degree))


# Euclidian Norm (or just length of vector)
def norm(v):
	return math.sqrt(dot_product(v, v))


def square(a):
	return dot_product(a, a)


def distance(v, w):
	return norm(vector_subtract(v, w))


# Using angle between vectors
def cross_product(v, w, alpha):
	alpha_degree = math.radians(alpha)
	return norm(v) * norm(w) * round(math.sin(alpha_degree))


def is_orthogonal(v, w):
	return True if dot_product(v, w) == 0 else False
