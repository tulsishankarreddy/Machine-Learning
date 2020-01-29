a = [2,3,4]
sum = 0
n = len(a)
for i in range(n):
    sum = sum + (((n - i) * (i + 1)) * a[i] )
print(sum)