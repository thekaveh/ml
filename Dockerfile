FROM quay.io/jupyter/datascience-notebook:python-3.11

WORKDIR /usr/src/app

COPY . .

# Issue #62 CPU image: no service startup and no source-built PyG extension.
RUN make install-torch-stack \
  && make nlp-assets \
  && python -m pip check \
  && python -m scripts.verify_torch_stack \
  && python -m scripts.verify_nnx_install
