# Better performance with tf.function
# To get peak performance and to make your model deployable anywhere, 
# use tf.function to make graphs out of your programs
# tf.function works best with TensorFlow ops, rather than NumPy ops or Python primitives
import tensorflow as tf

# Define a helper function to demonstrate the kinds of errors you might encounter:
import traceback
import contextlib

# Some helper code to demonstrate the kinds of errors you might encounter.
@contextlib.contextmanager
def assert_raises(error_class):
  try:
    yield
  except error_class as e:
    print('Caught expected exception \n  {}:'.format(error_class))
    traceback.print_exc(limit=2)
  except Exception as e:
    raise e
  else:
    raise Exception('Expected {} to be raised but no error was raised!'.format(
        error_class))

# Basics
@tf.function
def add(a, b):
  return a + b

add(tf.ones([2, 2]), tf.ones([2, 2]))  #  [[2., 2.], [2., 2.]]

v = tf.Variable(1.0)
with tf.GradientTape() as tape:
  result = add(v, 1.0)
tape.gradient(result, v)

# You can use functions inside functions
@tf.function
def dense_layer(x, w, b):
  return add(tf.matmul(x, w), b)

dense_layer(tf.ones([3, 2]), tf.ones([2, 2]), tf.ones([2]))


# Tracing and polymorphism
# Functions are polymorphic

@tf.function
def double(a):
  print("Tracing with", a)
  return a + a

print(double(tf.constant(1)))
print()
print(double(tf.constant(1.1)))
print()
print(double(tf.constant("a")))
print()

# control the tracing behavior
print("Obtaining concrete trace")
double_strings = double.get_concrete_function(tf.TensorSpec(shape=None, dtype=tf.string))
print("Executing traced function")
print(double_strings(tf.constant("a")))
print(double_strings(a=tf.constant("b")))
print("Using a concrete trace with incompatible types will throw an error")
with assert_raises(tf.errors.InvalidArgumentError):
  double_strings(tf.constant(1))

@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32),))
def next_collatz(x):
  print("Tracing with", x)
  return tf.where(x % 2 == 0, x // 2, 3 * x + 1)

print(next_collatz(tf.constant([1, 2])))
# We specified a 1-D tensor in the input signature, so this should fail.
with assert_raises(ValueError):
  next_collatz(tf.constant([[1, 2], [3, 4]]))

# Despite the multiple traces, the generated graph is actually identical, so this is a bit inefficient.
def train_one_step():
  pass

@tf.function
def train(num_steps):
  print("Tracing with num_steps = {}".format(num_steps))
  for _ in tf.range(num_steps):
    train_one_step()

train(num_steps=10)
train(num_steps=20)
# The simple workaround here is to cast your arguments to Tensors if they do not affect the shape of the generated graph.
train(num_steps=tf.constant(10))
train(num_steps=tf.constant(20))


# Side effects in tf.function
# Python side effects (like printing or mutating objects) only happen during tracing
# In general using a functional style will yield the best results.
@tf.function
def f(x):
  print("Traced with", x)
  tf.print("Executed with", x)

f(1)
f(1)
f(2)
# If you would like to execute Python code during each invocation of a tf.function, tf.py_function is an exit hatch
external_list = []

def side_effect(x):
  print('Python side effect')
  external_list.append(x)

@tf.function
def f(x):
  tf.py_function(side_effect, inp=[x], Tout=[])

f(1)
f(1)
f(1)
assert len(external_list) == 3
# .numpy() call required because py_function casts 1 to tf.constant(1)
assert external_list[0].numpy() == 1


# Beware of Python state
# To give one example, advancing iterator state is 
# a Python side effect and therefore only happens during tracing.
external_var = tf.Variable(0)
@tf.function
def buggy_consume_next(iterator):
  external_var.assign_add(next(iterator))
  tf.print("Value of external_var:", external_var)

iterator = iter([0, 1, 2, 3])
buggy_consume_next(iterator)
# This reuses the first value from the iterator, rather than consuming the next value.
buggy_consume_next(iterator)
buggy_consume_next(iterator)

# If you want to iterate over Python data, the safest way is to wrap it in a tf.data
def measure_graph_size(f, *args):
  g = f.get_concrete_function(*args).graph
  print("{}({}) contains {} nodes in its graph".format(
      f.__name__, ', '.join(map(str, args)), len(g.as_graph_def().node)))

@tf.function
def train(dataset):
  loss = tf.constant(0)
  for x, y in dataset:
    loss += tf.abs(y - x) # Some dummy computation.
  return loss

small_data = [(1, 1)] * 2
big_data = [(1, 1)] * 10
measure_graph_size(train, small_data)
measure_graph_size(train, big_data)

measure_graph_size(train, tf.data.Dataset.from_generator(
    lambda: small_data, (tf.int32, tf.int32)))
measure_graph_size(train, tf.data.Dataset.from_generator(
    lambda: big_data, (tf.int32, tf.int32)))


# Automatic control dependencies

a = tf.Variable(1.0)
b = tf.Variable(2.0)

@tf.function
def f(x, y):
  a.assign(y * b)
  b.assign_add(x * a)
  return a + b

f(1.0, 2.0)  # 10.0


# Variables

@tf.function
def f(x):
  v = tf.Variable(1.0)
  v.assign_add(x)
  return v

with assert_raises(ValueError):
  f(1.0)
#Non-ambiguous code is ok, though.
v = tf.Variable(1.0)

@tf.function
def f(x):
  return v.assign_add(x)

print(f(1.0))  # 2.0
print(f(2.0))  # 4.0

# You can also create variables inside a tf.function as long as we can prove 
# that those variables are created only the first time the function is executed.
class C:
  pass

obj = C()
obj.v = None

@tf.function
def g(x):
  if obj.v is None:
    obj.v = tf.Variable(1.0)
  return obj.v.assign_add(x)

print(g(1.0))  # 2.0
print(g(2.0))  # 4.0

# Variable initializers can depend on function arguments and on values of other variables
state = []
@tf.function
def fn(x):
  if not state:
    state.append(tf.Variable(2.0 * x))
    state.append(tf.Variable(state[0] * 3.0))
  return state[0] * x * state[1]

print(fn(tf.constant(1.0)))
print(fn(tf.constant(3.0)))


# Using AutoGraph

# Simple loop
@tf.function
def f(x):
  while tf.reduce_sum(x) > 1:
    tf.print(x)
    x = tf.tanh(x)
  return x

f(tf.random.uniform([5]))

# If you're curious you can inspect the code autograph generates.
def f(x):
  while tf.reduce_sum(x) > 1:
    tf.print(x)
    x = tf.tanh(x)
  return x

print(tf.autograph.to_code(f))


# AutoGraph: Conditionals

# Here is a function that checks if the resulting graph uses tf.cond:
def test_tf_cond(f, *args):
  g = f.get_concrete_function(*args).graph
  if any(node.name == 'cond' for node in g.as_graph_def().node):
    print("{}({}) uses tf.cond.".format(
        f.__name__, ', '.join(map(str, args))))
  else:
    print("{}({}) executes normally.".format(
        f.__name__, ', '.join(map(str, args))))

  print("  result: ",f(*args).numpy())

# Passing a python True executes the conditional normally:
@tf.function
def dropout(x, training=True):
  if training:
    x = tf.nn.dropout(x, rate=0.5)
  return x

test_tf_cond(dropout, tf.ones([10], dtype=tf.float32), True)

# But passing a tensor replaces the python if with a tf.cond:
test_tf_cond(dropout, tf.ones([10], dtype=tf.float32), tf.constant(True))

# Tracing both sides can result in unexpected execution of Python code.
@tf.function
def f(x):
  if x > 0:
    x = x + 1.
    print("Tracing `then` branch")
  else:
    x = x - 1.
    print("Tracing `else` branch")
  return x

f(-1.0).numpy()
f(1.0).numpy()
f(tf.constant(1.0)).numpy()

# It requires that if one branch creates a tensor used downstream, 
# the other branch must also create that tensor.
@tf.function
def f():
  if tf.constant(True):
    x = tf.ones([3, 3])
  return x

# Throws an error because both branches need to define `x`.
with assert_raises(ValueError):
  f()

# If you want to be sure that a particular section of control flow is never converted by autograph, then 
# explicitly convert the object to a python type so an error is raised instead:
@tf.function
def f(x, y):
  if bool(x):
    y = y + 1.
    print("Tracing `then` branch")
  else:
    y = y - 1.
    print("Tracing `else` branch")
  return y

f(True, 0).numpy()
f(False, 0).numpy()
with assert_raises(TypeError):
  f(tf.constant(True), 0.0)


# AutoGraph and loops

# If a loop is not converted, it will be statically unrolled
def test_dynamically_unrolled(f, *args):
  g = f.get_concrete_function(*args).graph
  if any(node.name == 'while' for node in g.as_graph_def().node):
    print("{}({}) uses tf.while_loop.".format(
        f.__name__, ', '.join(map(str, args))))
  elif any(node.name == 'ReduceDataset' for node in g.as_graph_def().node):
    print("{}({}) uses tf.data.Dataset.reduce.".format(
        f.__name__, ', '.join(map(str, args))))
  else:
    print("{}({}) gets unrolled.".format(
        f.__name__, ', '.join(map(str, args))))

# For loops
@tf.function
def for_in_range():
  x = 0
  for i in range(5):
    x += i
  return x

test_dynamically_unrolled(for_in_range)

@tf.function
def for_in_tfrange():
  x = tf.constant(0, dtype=tf.int32)
  for i in tf.range(5):
    x += i
  return x

test_dynamically_unrolled(for_in_tfrange)

@tf.function
def for_in_tfdataset():
  x = tf.constant(0, dtype=tf.int64)
  for i in tf.data.Dataset.range(5):
    x += i
  return x

test_dynamically_unrolled(for_in_tfdataset)

@tf.function
def while_py_cond():
  x = 5
  while x > 0:
    x -= 1
  return x

test_dynamically_unrolled(while_py_cond)

@tf.function
def while_tf_cond():
  x = tf.constant(5)
  while x > 0:
    x -= 1
  return x

test_dynamically_unrolled(while_tf_cond)

# Compare the following examples:
@tf.function
def while_py_true_py_break(x):
  while True:  # py true
    if x == 0: # py break
      break
    x -= 1
  return x

test_dynamically_unrolled(while_py_true_py_break, 5)

@tf.function
def buggy_while_py_true_tf_break(x):
  while True:   # py true
    if tf.equal(x, 0): # tf break
      break
    x -= 1
  return x

with assert_raises(TypeError):
  test_dynamically_unrolled(buggy_while_py_true_tf_break, 5)

@tf.function
def while_tf_true_tf_break(x):
  while tf.constant(True): # tf true
    if x == 0:  # py break
      break
    x -= 1
  return x

test_dynamically_unrolled(while_tf_true_tf_break, 5)

@tf.function
def buggy_py_for_tf_break():
  x = 0
  for i in range(5):  # py for
    if tf.equal(i, 3): # tf break
      break
    x += i
  return x

with assert_raises(TypeError):
  test_dynamically_unrolled(buggy_py_for_tf_break)

@tf.function
def tf_for_py_break():
  x = 0
  for i in tf.range(5): # tf for
    if i == 3:  # py break
      break
    x += i
  return x

test_dynamically_unrolled(tf_for_py_break)

# In order to accumulate results from a dynamically unrolled loop, 
# you'll want to use tf.TensorArray.
batch_size = 2
seq_len = 3
feature_size = 4

def rnn_step(inp, state):
  return inp + state

@tf.function
def dynamic_rnn(rnn_step, input_data, initial_state):
  # [batch, time, features] -> [time, batch, features]
  input_data = tf.transpose(input_data, [1, 0, 2])
  max_seq_len = input_data.shape[0]

  states = tf.TensorArray(tf.float32, size=max_seq_len)
  state = initial_state
  for i in tf.range(max_seq_len):
    state = rnn_step(input_data[i], state)
    states = states.write(i, state)
  return tf.transpose(states.stack(), [1, 0, 2])
  
dynamic_rnn(rnn_step,
            tf.random.uniform([batch_size, seq_len, feature_size]),
            tf.zeros([batch_size, feature_size]))


# Gotcha's

# Zero iterations
# Here is an example of incorrect code:
@tf.function
def buggy_loop_var_uninitialized():
  for i in tf.range(3):
    x = i
  return x

with assert_raises(ValueError):
  buggy_loop_var_uninitialized()

# And the correct version:
@tf.function
def f():
  x = tf.constant(0)
  for i in tf.range(3):
    x = i
  return x

f()

# Consistent shapes and types
# Here is an incorrect example that attempts to change a tensor's type:
@tf.function
def buggy_loop_type_changes():
  x = tf.constant(0, dtype=tf.float32)
  for i in tf.range(3): # Yields tensors of type tf.int32...
    x = i
  return x

with assert_raises(TypeError):
  buggy_loop_type_changes()

# Here is an incorrect example that attempts to change a Tensor's shape while iterating:
@tf.function
def buggy_concat():
  x = tf.ones([0, 10])
  for i in tf.range(5):
    x = tf.concat([x, tf.ones([1, 10])], axis=0)
  return x

with assert_raises(ValueError):
  buggy_concat()


@tf.function
def concat_with_padding():
  x = tf.zeros([5, 10])
  for i in tf.range(5):
    x = tf.concat([x[:i], tf.ones([1, 10]), tf.zeros([4-i, 10])], axis=0)
    x.set_shape([5, 10])
  return x

concat_with_padding()


