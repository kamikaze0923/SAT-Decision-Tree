import numpy as np

data = open("weather.csv")
feature_name = data.readline().strip('\n').split(",")
feature_cata = [[] for _ in range(len(feature_name))]

num_examples = 0
while True:
    line = data.readline().strip('\n').split(",")
    if line == ['']:
        data.close()
        break
    for i,f in enumerate(line):
        if f not in feature_cata[i]:
            feature_cata[i].append(f)
    num_examples += 1



print(feature_name)
print(feature_cata)

num_features = sum([len(i) for i in feature_cata[:-1]])
data_x = np.zeros(shape=(num_examples, num_features), dtype=np.int8)
data_y = np.zeros(shape=(num_examples), dtype=np.int8)

print(data_x.shape)

data = open("weather.csv")
data.readline()

n = 0
while True:
    line = data.readline().strip('\n').split(",")
    if line == ['']:
        data.close()
        break
    for i,f in enumerate(line[:-1]):
        prev = 0
        for j in range(0,i):
            prev += len(feature_cata[j])
        current = feature_cata[i].index(f) + prev
        data_x[n, current] = 1
    if line[-1] == feature_cata[-1][1]:
        data_y[n] = 1
    n += 1

print(data_x)
print(data_y)

np.save("x.npy", data_x)
np.save("y.npy", data_y)




