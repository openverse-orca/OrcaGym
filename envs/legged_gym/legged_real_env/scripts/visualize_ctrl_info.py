import pickle

with open('ctrl_info.pkl', 'rb') as f:
    data = pickle.load(f)


# data["Lite3_000"]["action_scale"] = 0.25


# with open('ctrl_info.pkl', 'wb') as f:
#     pickle.dump(data, f)

print(data)
