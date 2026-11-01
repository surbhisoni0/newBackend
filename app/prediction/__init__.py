import os as _os
# One thread per math library. On a small shared host (Render free = 0.1 CPU) the default "one thread per visible core" makes every model refit
# fight the sync loop and the web server for that sliver of CPU and stalls them. Results are unchanged (checked: identical picks). Must run before numpy/sklearn load.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
