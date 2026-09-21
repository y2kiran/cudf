ARG BASE_IMAGE=ubuntu:24.04
FROM $BASE_IMAGE
ARG DEBIAN_FRONTEND=noninteractive
ARG PARALLEL_LEVEL
WORKDIR /

# Set visible devices and mount NVIDIA driver binary utilities inside the
# container
#
# See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html#dockerfiles
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install common packages for development
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
    ninja-build \
    wget \
    sudo \
    gosu \
    git \
    vim \
    ccache \
    libpciaccess-dev \
    pciutils \
    ca-certificates \
    gnupg \
    file \
    pkg-config \
    binutils \
    binutils-dev \
    openssh-client \
    openmpi-bin \
    libopenmpi-dev \
    gcc-14 \
    g++-14 \
    gdb \
    sqlite3 \
    ncat \
    && rm -rf /var/lib/apt/lists/*

# Install Nsight Systems and Nsight Compute.
# Copy docker/tools/nsight_{compute,systems}-linux-{x86_64,arm}.{run,deb} into
# this repo before building (see docker/tools/README.md); they are gitignored.
COPY docker/tools/ /tmp/nsight-tools/
RUN set -ex \
    && NSIGHT_ARCH=$(uname -m | sed 's/aarch64/arm/') \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends "/tmp/nsight-tools/nsight_systems-linux-${NSIGHT_ARCH}.deb" \
    && sh "/tmp/nsight-tools/nsight_compute-linux-${NSIGHT_ARCH}.run" --quiet -- -noprompt -targetpath=/opt/nvidia/nsight-compute \
    && rm -rf /tmp/nsight-tools /var/lib/apt/lists/*

ENV PATH=${PATH}:/opt/nvidia/nsight-compute:/opt/nvidia/nsight-systems/bin

# Install Miniforge3
RUN wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" -O /miniforge.sh \
    && sh /miniforge.sh -b -p /conda \
    && rm /miniforge.sh \
    && /conda/bin/conda init bash --system

ENV PATH=${PATH}:/conda/bin
# Enables "source activate"
SHELL ["/bin/bash", "-c"]

# Compile libcudf from source.
WORKDIR /cudf
COPY . .
RUN git submodule update --init --recursive \
    && mamba env create -q --name cudf --file conda/environments/all_cuda-133_arch-$(uname -m).yaml \
    && source activate cudf \
    && PARALLEL_LEVEL=${PARALLEL_LEVEL:-$(nproc)} CUDF_CMAKE_CUDA_ARCHITECTURES="80-real;90-real;100-real;120" ./build.sh libcudf tests benchmarks --ptds --cmake-args=\"-DCUDF_ENABLE_ARROW_S3=OFF -DCUDA_ENABLE_LINEINFO=ON\" \
    && conda clean --all

# Activate the conda environment when launching the container
RUN echo "source activate cudf" >> ~/.bashrc
