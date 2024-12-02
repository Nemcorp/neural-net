import math

inputs = [(0.0000, 0.0000), (0.1600, 0.1556), (0.2400, 0.3543), (0.2800, 0.3709),
    (0.3600, 0.4702), (0.4000, 0.4868), (0.5000, 0.5530), (0.5200, 0.6026),
    (0.6000, 0.6358), (0.6200, 0.3212), (0.6600, 0.7185), (0.7000, 0.7351),
    (0.7600, 0.8013), (0.8400, 0.7848), (0.9600, 0.9669), (1.0000, 1.0000)]
targets = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]

test_inputs = [(0.1600, 0.1391), (0.5600, 0.3046),
    (0.7600, 0.8013), (0.9600, 0.3046), (0.1600, 0.7185)]
test_targets = [0, 0, 1, 0, 0]


weights = [.1,.2]
b = .3
learning_rate = .5
epochs = 4000

def predict(inp):
    return sum([i*w for i, w in zip(inp, weights)]) + b

def log_loss(act, target) :
    return (-target * math.log(act)) - ((1-target) * math.log(1-act))

def activate(x):
    return 1 / (1+ math.exp(-x))


#training loop
for _ in range(epochs):
    #feed forward
    preds = [predict(inp) for inp in inputs]
    activations = [activate(p) for p in preds]

    #back prop
    errors_d = [(a - t) for a, t in zip(activations, targets)]
    weights_d = [[e* i for i in inp] for inp, e in zip(inputs, errors_d)]
    translated_weights_d = list(zip(*weights_d))

    cost = sum([log_loss(act, t) for act,t in zip(activations, targets)]) / len(activations)
    bias = errors_d

    for i in range(len(weights)) :
        # update weights
        weights[i] -= learning_rate * sum(translated_weights_d[i]) / len(translated_weights_d[i])
    b -= learning_rate * sum(bias) / len(bias)

    #print
    print(f"Epoch: {epochs}, Cost: {cost:.2f}")

testPreds = [predict(inp) for inp in test_inputs]
testActivations = [activate(p) for p in testPreds]
for t, a in zip(test_targets, testActivations):
    print(f"Target: {t:.1f}, Prediction: {a:.1f}")



# inputs = [(0.0000, 0.0000), (0.1600, 0.1556), (0.2400, 0.3543), (0.2800, 0.3709),
#     (0.3600, 0.4702), (0.4000, 0.4868), (0.5000, 0.5530), (0.5200, 0.6026),
#     (0.6000, 0.6358), (0.6200, 0.3212), (0.6600, 0.7185), (0.7000, 0.7351),
#     (0.7600, 0.8013), (0.8400, 0.7848), (0.9600, 0.9669), (1.0000, 1.0000)]
# targets = [230, 555, 815, 860, 1140, 1085, 1200, 1330, 1290,
#     870, 1545, 1480, 1750, 1845, 1790, 1955]

# test_inputs = [(0.1600, 0.1391), (0.5600, 0.3046),
#     (0.7600, 0.8013), (0.9600, 0.3046), (0.1600, 0.7185)]
# test_targets = [500, 850, 1650, 950, 1375]



# weights = [.1,.2]
# b = .1
# learning_rate = .1
# epochs = 30000

# def predict(inp) :
#     return sum([i * w for i, w in zip(inp, weights)]) + b

# #training loop
# for _ in range(epochs) :
#     # feed forward
#     preds = [predict(inp) for inp in inputs]
#     errors_d = [2*(p - t) for p, t in zip(preds, targets)]
#     weights_d = [[(e * i) for i in inp] for e, inp in zip(errors_d, inputs)]
#     transposed_weights = list(zip(*weights_d))

#     bias = errors_d
#     cost = sum([(p-t)**2 for p, t in zip(preds, targets)]) / len(targets)

#     # back prop
#     for i in range(len(weights)):
#         weights[i] -= learning_rate * sum(transposed_weights[i]) / len(transposed_weights[i])
#     b -=  learning_rate * sum(bias) / len(bias)

#     # print
#     print(f"Epoch: {epochs}, Cost: {cost:.2f}")



# weights = [.1, .2]
# learning_rate = .1
# b = .1
# epochs = 5000

# def predict(inputs):
#     return sum([i*w for i,w in zip(inputs, weights)]) + b

# #training loop
# for _ in range(epochs):
#     #feed forward
#     preds = [predict(inp) for inp in inputs]
#     errors_d = [2*(p-t) for p, t in zip(preds, targets)]
#     weights_d = [[e*i for i in inp] for e, inp in zip(errors_d, inputs)]
#     cost = sum([(p-t)**2 for p,t in zip(preds, targets)]) / len(targets)
#     transposed_weights = list(zip(*weights_d))
#     bias = errors_d

#     for i in range(len(weights)):
#         weights[i] -= learning_rate * (sum(transposed_weights[i]) / len(transposed_weights))
#     b -= learning_rate * sum(bias) / len(bias)

#     print(f"Epochs: {epochs}, Cost: {cost:.1f}")

# testPreds = [predict(inp) for inp in test_inputs]

# for t, p in zip(test_targets, testPreds):
#     print(f"Target: {t:.1f}, Prediction: {p:.1f}")





# w1 = .1
# w2 = .2
# b = .1
# learning_rate = .1
# epochs = 5000

# def predict(i1, i2):
#     return i1 *w1 + i2 *w2 + b

# #training loop
# for _ in range(epochs):
#     #feed forward
#     preds = [predict(i1, i2) for i1, i2 in inputs]
#     cost = sum([(p-t)**2 for p,t in zip(preds, targets)]) / len(targets)
#     errors_d = [2*(p-t) for p, t in zip(preds, targets)]
#     weights1_d = [e*i[0] for e, i in zip(errors_d, inputs)]
#     weights2_d = [e*i[1] for e, i in zip(errors_d, inputs)]
#     bias = errors_d


#     b -= learning_rate * (sum(bias) / len(bias))
#     w1 -= learning_rate * (sum(weights1_d) / len(weights1_d))
#     w2 -= learning_rate * (sum(weights2_d) / len(weights2_d))

#     print(f"Epochs: {epochs}, Cost: {cost:.1f}")

# testPreds = [predict(i1, i2) for i1, i2 in test_inputs]

# for t, p in zip(test_targets, testPreds):
#     print(f"Target: {t:.1f}, Prediction: {p:.1f}")



# w1 = .1
# w2 = .2
# b = .1
# learning_rate = .1
# epochs = 300

# def predict(i1, i2) :
#     return i1 * w1 + w2*i2 + b

# #training loop
# for _ in range(epochs) :
#     #feed forward
#     preds = [predict(i1, i2) for i1, i2 in inputs]
#     errors_d = [2*(p-t) for p, t in zip(preds, targets)]
#     weights1_d = [e * i[0] for e,i in zip(errors_d, inputs)]
#     weights2_d = [e * i[1] for e,i in zip(errors_d, inputs)]
#     bias = errors_d

#     cost = sum([(p-t) ** 2 for p, t in zip(preds, targets)]) / len(inputs)

#     #back prop
#     w1 -= learning_rate * (sum(weights1_d) / len(weights1_d))
#     w2 -= learning_rate * (sum(weights2_d) / len(weights2_d))
#     b -= learning_rate * (sum(bias) / len(bias))

#     print(f"Epoch: {epochs}, Cost: {cost:.2f}")

# print(f"Weight1: {w1:.2f}, Weight2: {w2:.2f},Bias: {b:.2f}")




# inputs = [1,2,3,4]
# targets = [2,4,6,8]

# w = .1
# b = .3
# learning_rate = .1

# epochs = 100


# def predict(i) :
#     return w * i + b


# # training run for one neuron
# for _ in range(epochs) :
#     # feed forward
#     preds = [predict(i) for i in inputs]

#     # back propagation
#     errors_d = [2*(p-t) for p, t in zip(preds, targets)]
#     weights_d = [e*i for e, i in zip(errors_d, inputs)]
#     bias_d = errors_d

#     cost = sum(weights_d) / len(weights_d)
#     w -= learning_rate * cost
#     b -= learning_rate * (sum(bias_d) / len(bias_d))

#     # output
#     print(f"Weight: {w:.2f}, Cost: {cost:.2f}, Bias: {b:.2f}")





# inputs = [1.1,2.1,3,4]
# targets = [11,14.2,16.5,17.1]
# epochs = 10000

# w = .1 # the slope
# b = 0.3
# learning_rate = .11

# def predict(i):
#     return w * i + b


# #train the network
# for _ in range(epochs):

#     pred = [predict(i) for i in inputs]
#     # simple cost function:
#     # errors = [t - p for p, t in zip(pred, targets)]
#     # cost = sum(errors) / len(targets)
#     # mean squared cost function:

#     errors_d = [2 * (p-t) for p, t in zip(pred, targets)]
#     weights_d = [e * i for e, i in zip(errors_d, inputs)]
#     bias_d = [e*1 for e in errors_d]
    
    
#     cost = sum(weights_d) / len(weights_d)

#     print(f"Weight: {w:.2f}, Cost: {cost:.2f}, Bias: {b:.2f}")

#     w-= learning_rate * cost
#     b-= (sum(bias_d) /len(bias_d)) * learning_rate

# # test the network
# test_inputs = [5,6]
# test_targets = [20,22]
# pred = [predict(i) for i in test_inputs]
# for i, t, p in zip(test_inputs, test_targets, pred):
#     print(f"input:{i}, target:{t}, pred:{p:.4f}")





