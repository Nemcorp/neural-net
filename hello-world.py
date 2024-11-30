inputs = [(0.0000, 0.0000), (0.1600, 0.1556), (0.2400, 0.3543), (0.2800, 0.3709),
    (0.3600, 0.4702), (0.4000, 0.4868), (0.5000, 0.5530), (0.5200, 0.6026),
    (0.6000, 0.6358), (0.6200, 0.3212), (0.6600, 0.7185), (0.7000, 0.7351),
    (0.7600, 0.8013), (0.8400, 0.7848), (0.9600, 0.9669), (1.0000, 1.0000)]
targets = [230, 555, 815, 860, 1140, 1085, 1200, 1330, 1290,
    870, 1545, 1480, 1750, 1845, 1790, 1955]

w1 = 0.1
w2 = 0.5
epochs = 4000
learning_rate = 0.15
bias = 1

def predict(i1, i2):
    return w1 * i1  + w2 * i2 + bias

# train the network

for epoch in range(epochs):
    pred = [predict(i1, i2) for i1, i2 in inputs] 
    errors = [(p-t) ** 2 for p, t in zip(pred, targets)]
    cost = sum(errors) / len(targets)  # single number that indicates how well the network is doing
    print(f"Epoch: {epoch} Cost: {cost:.2f}")

    errors_derivative = [2 * (p - t) for p, t in zip(pred, targets)]
    weight1_delta = [e * i[0] for e, i in zip(errors_derivative, inputs)]
    weight2_delta = [e * i[1] for e, i in zip(errors_derivative, inputs)]
    bias_delta = [e * 1 for e in errors_derivative]
    
    w1 -= learning_rate * sum(weight1_delta) / len(weight1_delta)
    w2 -= learning_rate * sum(weight2_delta) / len(weight2_delta)
    bias -= learning_rate * sum(bias_delta) /len(bias_delta)


print(f"weight1:{w1: .4f}, weight2:{w2: .4f}, bias: {bias: .4f}")

# test the network
# test_inputs = [5, 6]
# test_targets = [20, 22]

# #pred = [predict(i) for i in test_inputs]

# for i, t, p in zip(test_inputs, test_targets, pred):
# print(f"input:{i}, target:{t}, prediction: {p: .4f}")

