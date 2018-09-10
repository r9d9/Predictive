memo = {1:3}
def recufu3(n):
    """returns 3*n"""
    if not n in memo:
        memo[n] = recufu3(n-1) + 3
    return memo[n]

print(recufu3(13))


def recusum(n):
    """returns sum of first n integers"""
    if n == 0:
        return 0
    else:
        return n + recusum(n-1)

print(recusum(10))
def pascal(n):
    """#https://www.python-course.eu/recursive_functions.php"""
    if n == 1:
        return [1]
    else:
        line = [1]
        previous_line = pascal(n-1)
        for i in range(len(previous_line)-1):
            line.append(previous_line[i] + previous_line[i+1])
        line += [1]
    return line

print(pascal(6))