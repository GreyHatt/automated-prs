def hello_world():
    print("Hello, World!")
    return "Hello, World!"

def add_numbers(a, b):
    """Function to add two numbers"""
    return a + b

def add_numbers_2(a, b):
    """Function to add two numbers and print"""
    result = a + b
    print(f"Result: {result}")
    return result


def subtract_numbers(a, b):
    """Function to subtract two numbers"""
    return a - b

if __name__ == "__main__":
    hello_world()
    result = add_numbers(3, 5)
    print(f"Addition Result: {result}")
    result2 = add_numbers_2(3, 5)
    print(f"Addition Result: {result2}")
    subtraction_result = subtract_numbers(10, 4)
    print(f"Subtraction Result: {subtraction_result}")