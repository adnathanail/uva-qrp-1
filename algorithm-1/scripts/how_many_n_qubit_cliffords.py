# A simple method for sampling random Clifford operators
# https://arxiv.org/abs/2008.06011
# Ewout van den Berg

# How to efficiently select an arbitrary Clifford group element
# Robert Koenig, John A. Smolin
# https://arxiv.org/abs/1406.2170


def clifford_group_size(n):
    result = 2 ** (n**2 + 2 * n)
    for k in range(1, n + 1):
        result *= 4**k - 1
    return result


if __name__ == "__main__":
    for i in range(1, 11):
        print(f"There are {clifford_group_size(i)} {i}-qubit Cliffords")
