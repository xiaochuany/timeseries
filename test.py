import time 

s = time.time()
for _ in range(10000):
	pass

e = time.time()
print(e-s)
