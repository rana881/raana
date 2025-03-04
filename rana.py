i1, i2 = 0.05, 0.10
w1, w2, w3, w4 = 0.15, 0.20, 0.25, 0.30
w5, w6, w7, w8 = 0.40, 0.45, 0.50, 0.55
b1, b2 = 0.35, 0.60
t1, t2 = 0.01, 0.99

def sigmoid(x):
    return 1 / (1 + (2.718 ** -x))

def sigmoid_derivative(x):
    return x * (1 - x)

net_h1 = i1 * w1 + i2 * w2 + b1
net_h2 = i1 * w3 + i2 * w4 + b1

h1 = sigmoid(net_h1)
h2 = sigmoid(net_h2)

net_o1 = h1 * w5 + h2 * w6 + b2
net_o2 = h1 * w7 + h2 * w8 + b2

o1 = sigmoid(net_o1)
o2 = sigmoid(net_o2)

error1 = 0.5 * (t1 - o1) ** 2
error2 = 0.5 * (t2 - o2) ** 2
total_error = error1 + error2

print(f'Forward Propagation Output: o1 = {o1}, o2 = {o2}, Total Error = {total_error}')

dE_total_o1 = -(t1 - o1)
dE_total_o2 = -(t2 - o2)

d_o1_net_o1 = sigmoid_derivative(o1)
d_o2_net_o2 = sigmoid_derivative(o2)

d_net_o1_w5 = h1
d_net_o1_w6 = h2
d_net_o2_w7 = h1
d_net_o2_w8 = h2

dE_total_w5 = dE_total_o1 * d_o1_net_o1 * d_net_o1_w5
dE_total_w6 = dE_total_o1 * d_o1_net_o1 * d_net_o1_w6

dE_total_w7 = dE_total_o2 * d_o2_net_o2 * d_net_o2_w7
dE_total_w8 = dE_total_o2 * d_o2_net_o2 * d_net_o2_w8

learning_rate = 0.5

w5 -= learning_rate * dE_total_w5
w6 -= learning_rate * dE_total_w6
w7 -= learning_rate * dE_total_w7
w8 -= learning_rate * dE_total_w8

print(f'Updated weights: w5 = {w5}, w6 = {w6}, w7 = {w7}, w8 = {w8}')