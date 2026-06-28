#!/usr/bin/env python3
"""
Command-line utility to run compile or test actions.

This script accepts a single positional argument with two possible values:
'compile' or 'test'. Based on the argument, it executes the corresponding
function, which can be customized to run system commands.
"""

import argparse
import os
import subprocess
from time import time


def cuda_compile() -> None:
    """Run the compile process.

    Modify this function to include the actual compile command and any
    required environment variables.
    """
    def compile_test(test_name: str) -> None:
        process = subprocess.Popen(
            f"nvcc test_suite/{test_name}.cu -o bin/{test_name}.o \
                    -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lX11 -lpng -lz".split(),
            # env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        print(f"Compiling: test_suite/{test_name}.cu")
        stdout, stderr = process.communicate()

        if stdout.strip():
            print(stdout)

        if stderr:
            print("Compile stderr:")
            print(stderr)

    os.makedirs('bin', exist_ok=True)

    # Compilations
    for test in os.listdir("test_suite"):
        compile_test(test.replace(".cu", ""))


def cuda_naive_test() -> None:
    """Run the test process.

    Modify this function to include the actual test command and any
    required environment variables.
    """

    print(f"Testing: cuda_naive")
    print(40*'=')
    for i in range(10):
        process = subprocess.Popen(
            f"./bin/cuda_naive.o".split(),
            # env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        begin = time()
        stdout, stderr = process.communicate()
        end = time()

        print(f"Ejecucion {i} realizada en {end-begin:.2f} segundos")

        if stderr:
            print("Test stderr:")
            print(stderr)
    print(40*'=')


def cuda_streams_test() -> None:
    """Run the test process.

    Modify this function to include the actual test command and any
    required environment variables.
    """

    print(f"Testing: cuda_streams")
    print(40*'=')
    for i in range(10):
        process = subprocess.Popen(
            f"./bin/cuda_streams.o".split(),
            # env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        begin = time()
        stdout, stderr = process.communicate()
        end = time()

        print(f"Ejecucion {i} realizada en {end-begin:.2f} segundos")

        if stderr:
            print("Test stderr:")
            print(stderr)
    print(40*'=')


def parallel_test(upper_limit: int = 11) -> None:
    env: dict[str, str] = os.environ.copy()

    stdout, stderr = subprocess.Popen(["./bin/get_num_threads"],
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE).communicate()
    THREAD_NUM = int(stdout)

    directory = "bin/parallel"
    for test_bin in os.listdir(directory):
        print(f"Testing: {test_bin}")
        print(40*'=')
        for i in range(8, upper_limit+1):
            N = 1 << i
            print(f"N = {N}")
            for thread_num in [1]+list(i for i in range(4, THREAD_NUM+1, 4)):
                print(f"threads = {thread_num} ")
                env["OMP_NUM_THREADS"] = str(thread_num)

                for j in range(1, 9):
                    base_case = 1 << j
                    # Evitar casos donde los casos bases son mas grandes que la dimension de la matriz
                    if base_case >= N:
                        break
                    print(f"base_case = {base_case} ")

                    for _ in range(10):
                        process = subprocess.Popen(
                            f"./{directory}/{test_bin} {N} {base_case}".split(),
                            env=env,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                        )

                        stdout, stderr = process.communicate()

                        print(stdout, end='')

                        if stderr:
                            print("Test stderr:")
                            print(stderr)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments with the selected action.
    """
    parser = argparse.ArgumentParser(
        description="Run compile or test actions.")
    parser.add_argument(
        "action",
        choices=["compile", "cuda_naive", "cuda_streams", "validate", "profile"],
        help="Action to perform: 'compile', 'cuda_naive', 'cuda_stream', 'validate' or 'profile'.",
    )

    return parser.parse_args()


def validate(upper_limit: int = 11) -> None:
    """Run the test process.

    Modify this function to include the actual test command and any
    required environment variables.
    """

    directory = "bin"
    print("Validating algorithms.")
    print(40*'=')

    N = 1<< upper_limit

    process = subprocess.Popen( f"./{directory}/alg_validation {N}".split())

    stdout, stderr = process.communicate()

    if stderr:
        print("Test stderr:")
        print(stderr)
    print(40*'=')


def profile() -> None:
    pass


def main() -> None:
    """Main entry point of the script."""
    begin = time()
    args = parse_args()

    if args.action == "compile":
        cuda_compile()
    elif args.action == "cuda_naive":
        cuda_naive_test()
    elif args.action == "cuda_streams":
        cuda_streams_test()
    elif args.action == "validate":
        validate()
    elif args.action == "profile":
        profile()
    else:
        # This branch should not be reached due to argparse choices
        raise ValueError(f"Unsupported action: {args.action}")

    end = time()
    print(f"Ejecucion realizada en {end-begin:.2f} segundos")


if __name__ == "__main__":
    main()
