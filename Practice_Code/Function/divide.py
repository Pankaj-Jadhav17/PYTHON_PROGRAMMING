num3 = float(input("Enter the divident for division: "))
num4 = float(input("Enter the divisor for division: "))
if num4 == 0:
    print("Error: Division by zero is not allowed.")
else:
    result_division = num3 / num4
    print(f"Division: {num3} / {num4} = {result_division}")