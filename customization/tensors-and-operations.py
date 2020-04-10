# A Tensor is a multi-dimensional array. 
# Similar to NumPy ndarray objects, tf.
# Tensor objects have a data type and a shape. 
# Additionally, tf.Tensors can reside in accelerator memory (like a GPU). 
# TensorFlow offers a rich library of operations 
# (tf.add, tf.matmul, tf.linalg.inv etc.) that consume and produce tf.Tensors.

import tensorflow as tf
print(tf.add(1, 2))
print(tf.add([1, 2], [3, 4]))
print(tf.square(5))
print(tf.reduce_sum([1, 2, 3]))

# Operator overloading is also supported
print(tf.square(2) + tf.square(3))

x = tf.matmul([[1]], [[2, 3]])
print(x)
print(x.shape)
print(x.dtype)


# Converting between a TensorFlow tf.Tensors and a NumPy ndarray is easy:
# TensorFlow operations automatically convert NumPy ndarrays to Tensors.
# NumPy operations automatically convert Tensors to NumPy ndarrays.
import numpy as np

ndarray = np.ones([3, 3])

print("TensorFlow operations convert numpy arrays to Tensors automatically")
tensor = tf.multiply(ndarray, 42)
print(tensor)


print("And NumPy operations convert Tensors to numpy arrays automatically")
print(np.add(tensor, 1))

print("The .numpy() method explicitly converts a Tensor to a numpy array")
print(tensor.numpy())


# Many TensorFlow operations are accelerated using the GPU for computation. 
# Without any annotations, TensorFlow automatically decides whether to use 
# the GPU or CPU for an operation—copying the tensor between CPU and GPU memory, 
# if necessary. Tensors produced by an operation are typically backed by 
# the memory of the device on which the operation executed

x = tf.random.uniform([3, 3])

print("Is there a GPU available: "),
print(tf.config.experimental.list_physical_devices("GPU"))

print("Is the Tensor on GPU #0:  "),
print(x.device.endswith('GPU:0'))


# In TensorFlow, placement refers to how individual operations 
# are assigned (placed on) a device for execution. 
# As mentioned, when there is no explicit guidance provided, 
# TensorFlow automatically decides which device to execute an operation 
# and copies tensors to that device, if needed. 
# However, TensorFlow operations can be explicitly placed 
# on specific devices using the tf.device context manager
import time

def time_matmul(x):
  start = time.time()
  for loop in range(10):
    tf.matmul(x, x)

  result = time.time()-start

  print("10 loops: {:0.2f}ms".format(1000*result))

# Force execution on CPU
print("On CPU:")
with tf.device("CPU:0"):
  x = tf.random.uniform([1000, 1000])
  assert x.device.endswith("CPU:0")
  time_matmul(x)

# Force execution on GPU #0 if available
if tf.config.experimental.list_physical_devices("GPU"):
  print("On GPU:")
  with tf.device("GPU:0"): # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
    x = tf.random.uniform([1000, 1000])
    assert x.device.endswith("GPU:0")
    time_matmul(x)



# Create a source dataset using one of the factory functions like 
# Dataset.from_tensors, Dataset.from_tensor_slices, 
# or using objects that read from files like TextLineDataset or TFRecordDataset. 
# See the TensorFlow Dataset guide for more information
ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])

# Create a CSV file
import tempfile
_, filename = tempfile.mkstemp()

with open(filename, 'w') as f:
  f.write("""Line 1
Line 2
Line 3
  """)

ds_file = tf.data.TextLineDataset(filename)
# Use the transformations functions like map, batch, 
# and shuffle to apply transformations to dataset records.
ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)
ds_file = ds_file.batch(2)
# tf.data.Dataset objects support iteration to loop over records:
print('Elements of ds_tensors:')
for x in ds_tensors:
  print(x)

print('\nElements in ds_file:')
for x in ds_file:
  print(x)
