import os

if not os.environ.get("OMP_NUM_THREADS"):
    os.environ("OMP_NUM_THREADS") = 1

    