
def clip(x, min, max):
    if x < min:
        return min
    if x > max:
        return max
    return x

def sign(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0

def sign_positive(x):
    if x >= 0:
        return 1
    return -1

# Unsure what the proper name of this operation is
def modulo_star(x, R):
    return ((x + 2**(R-1)) % 2**R) - 2**(R-1)
