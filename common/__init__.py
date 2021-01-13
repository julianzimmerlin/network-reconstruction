import utils
import sys

print(sys.maxsize)

mat = utils.id2matrix(1000000818, 10)
mat2 = utils.id2matrix(1000000818, 10)

print(utils.hash_tensor(mat))
print(utils.hash_tensor(mat2))
