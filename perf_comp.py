import time
from Tasks import task3, task5

def measure_time(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return end_time - start_time, result

def main():
    starting_point = ...  # Define your starting point here

    time_task3, result_task3 = measure_time(task3, starting_point)
    time_task5, result_task5 = measure_time(task5, starting_point)

    print(f"task3 took {time_task3:.6f} seconds")
    print(f"task5 took {time_task5:.6f} seconds")

if __name__ == "__main__":
    main()