limit = int(input("Enter the limit: "))

sum = 0
for i in range(1, limit + 1):
    sum += i
    print("the sum of first", i, "natural numbers is:", sum)
    