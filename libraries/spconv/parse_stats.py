import pickle
import torch
torch.set_printoptions(profile="full")

with open('debug.txt', 'rb') as f:
    data_list = []
    before = None
    before2 = None
    pair = None
    i=0
    while True:
        try:
            data = pickle.load(f)
        except EOFError:
            break
        if isinstance(data[1], torch.Tensor):
            print(data[1].shape)
        print(data[0], data[1])
        # if "pair_mask_fwd: "==data[0]:
        #     print("SORTED: ", torch.sort(data[1], descending=True)) 
        # if "subm pair_mask[0]: " == data[0]:
        #     pair = torch.zeros(data[1].shape)
        #     pair[data[1]==8192]=1
        #     print("pair shape: ", pair.shape)
        #     before = torch.argsort(data[1])
        # elif "subm pair[0]: " == data[0]:
        #     a = torch.zeros(data[1].shape)
        #     a[data[1]==-1] = 1
        #     b = torch.count_nonzero(a, dim=0)
        #     c = torch.zeros(b.shape)
        #     c[b==27]=1
        #     print("c shape: ", c.shape)
        #     print("SAME?: ", torch.equal(c, pair))
        # elif "subm argsort: " == data[0]:
        #     print("SSS? ", before.to("cuda:0") == data[1]) 
        if "subm argsort: " == data[0]:
            a = torch.arange(0, data[1].shape[1], device="cuda:0").view(1, data[1].shape[1])
            print(a.shape)
            print("SAME?: ", torch.equal(a, data[1]))
        print("\n")
