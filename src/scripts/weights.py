import math

# x = [17, 162, 2265, 50, 2680, 1297, 250, 354, 454, 44, 1094, 2891, 127, 312, 1395, 1815, 911, 1837]
x = [55, 25]

x.reverse()

total = sum(x)

for v in x:
    w = math.sqrt(total/(len(x) * v))
    # w = (total/(len(x) * v))

    print("\t- type: value")
    print(f"\t  value: {w}")
