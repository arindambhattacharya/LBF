import dpbf


dpbf.init_bloom_filter(1024,3,1,30,0.0001)
for i in range(0,10000):
	dpbf.insert(i)


dpbf.update()
print(dpbf.check(1))
for i in range(10):
	print (dpbf.check(i))
print(dpbf.getMemory())
print(dpbf.getMapSize())
print(dpbf.getTreeSize())
