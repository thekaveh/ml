FROM quay.io/jupyter/datascience-notebook:python-3.11@sha256:14379f24c840f27375e3a4b29a9fa55e449633ac85c9cf3806ca62d11e5603ec

WORKDIR /usr/src/app

COPY . .

# Keep the reviewed environment separate from packages preinstalled by the Jupyter base.
ENV VIRTUAL_ENV=/home/jovyan/.venvs/ml-eng-lab
ENV CONDA_AUTO_ACTIVATE_BASE=false
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Issue #63 locked CPU image: no service startup and no source-built PyG extension.
RUN /opt/conda/bin/python -m venv "$VIRTUAL_ENV" \
  && make install-torch-stack \
  && make nlp-assets \
  && python -m pip check \
  && python -m scripts.verify_torch_stack \
  && python -m scripts.verify_nnx_install
