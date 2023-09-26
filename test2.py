v = 0
for i in range(100):
    v0 = 1+0.9*v
    v1 = -1 + 0.9*v
    v2 = 0.9*v
    v = v0
    print(i, v0, v1, v2)