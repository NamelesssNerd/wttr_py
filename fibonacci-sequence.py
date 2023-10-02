def generate_fibonacci(n):
    """
    Generate a Fibonacci sequence up to the nth term.
    :param n: The number of terms to generate.
    :return: A list containing the Fibonacci sequence.
    """
    fibonacci_sequence = [0, 1]

    while len(fibonacci_sequence) < n:
        next_term = fibonacci_sequence[-1] + fibonacci_sequence[-2]
        fibonacci_sequence.append(next_term)

    return fibonacci_sequence

if __name__ == "__main__":
    n = int(input("Enter the number of Fibonacci terms to generate: "))

    if n <= 0:
        print("Please enter a positive integer.")
    else:
        fibonacci_sequence = generate_fibonacci(n)
        print(f"Fibonacci Sequence (up to {n} terms): {fibonacci_sequence}")