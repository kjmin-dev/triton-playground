FROM nvcr.io/nvidia/tritonserver:24.05-py3

ARG TRITON_RUNTIME_PROFILE=baseline

COPY packages/triton/runtime-requirements.txt /tmp/runtime-requirements.txt

RUN if [ "$TRITON_RUNTIME_PROFILE" = "localize" ]; then \
      apt-get update && apt-get install -y --no-install-recommends libsndfile1 && \
      rm -rf /var/lib/apt/lists/* && \
      python3 -m pip install --no-cache-dir --upgrade pip && \
      python3 -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu124 "torch>=2.6" && \
      python3 -m pip install --no-cache-dir -r /tmp/runtime-requirements.txt; \
    fi
