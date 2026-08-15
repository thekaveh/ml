FROM quay.io/jupyter/datascience-notebook:python-3.11

WORKDIR /usr/src/app

COPY . .

# The canonical Torch 2.11 runtime uses three PyG wheels and importable
# torchao 0.18. The complete quantization notebook stays manual-only under
# Issue #66; this build installs no service runtime.
RUN make install-torch-stack \
  && make nlp-assets \
  && python -m pip check \
  && python -m scripts.verify_torch_stack \
  && python -m scripts.verify_nnx_install
