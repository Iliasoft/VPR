import numpy as np
'''
digits = np.array([1, 3, 5, 7, 9, 11, 13, 15])
x1, x2, x3 = 0, 0, 0
while x1 + x2 + x3 != 30:
    x1, x2, x3 = np.random.choice(digits, replace=False), np.random.choice(digits, replace=False), np.random.choice(digits, replace=False)

print(x1, x2, x3)
'''
from tqdm import tqdm

data = [{6, 7, 8, 9, 10}, {10, 11, 12, 13, 14}, {14, 15, 5}, {1, 2, 3, 4, 5}]
#6: [7, 8, 9, 10] + 10: [11, 12, 13, 14] + 14: [15, 5]
# 1: [2, 3, 4, 5] + #6: [7, 8, 9, 10] + 10: [11, 12, 13, 14] + 14: [15, 5]

def flatten(data):

    completed = False
    while not completed:
        completed = True
        #counter = 0
        for item_array_id in tqdm(range(len(data))):
            for inner_item_array_id in range(item_array_id, len(data)):
                if item_array_id == inner_item_array_id:
                    continue

                if len(data[item_array_id] & data[inner_item_array_id]):
                    # print(item_array_id, inner_item_array_id)
                    data[item_array_id].update(data[inner_item_array_id])
                    data[inner_item_array_id] = set()
                    completed = False
                    #counter += 1
        # print(counter)
        new_list = []
        for item in data:
            if len(item):
                new_list.append(item)
        data = new_list
    return data

# print(flatten(data))

