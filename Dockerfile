FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

ENV CUDA_HOME=/usr/local/cuda-11.8
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"


RUN apt-get update && apt-get install -y \
    build-essential cmake curl ffmpeg git \
    libassimp-dev libavcodec-dev libavdevice-dev \
    libboost-all-dev libcgal-dev libceres-dev \
    libeigen3-dev libembree-dev libflann-dev \
    libfreeimage-dev libgflags-dev libglew-dev \
    libglfw3-dev libgoogle-glog-dev libgtk-3-dev \
    liblz4-dev libmetis-dev libopencv-dev \
    libqt5opengl5-dev libsqlite3-dev libsuitesparse-dev \
    libxxf86vm-dev nano neovim \
    ninja-build wget cuda-toolkit \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone https://github.com/colmap/colmap.git && \
    cd colmap && \
    git checkout 3.8 && \
    mkdir build && cd build && \
    cmake .. -GNinja \
        -DCUDA_ENABLED=ON \
        -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86;89" && \
    ninja install


RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" && \
    bash Miniforge3-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniforge3-Linux-x86_64.sh

ENV PATH=/opt/conda/bin:/usr/local/bin:$PATH

WORKDIR /app
COPY . .

RUN cd /app/SIBR_viewers && \
    cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j$(nproc) --target install

RUN git clone https://github.com/nv-tlabs/Difix3D.git
ENV PYTHONPATH=/app/Difix3D/src

RUN conda env create -f sam_environment.yaml
RUN conda run -n sam3 pip install "git+https://github.com/facebookresearch/sam3.git"
RUN conda run -n sam3 pip install torch torchvision --index-url "https://download.pytorch.org/whl/cu124"

RUN conda env create -f environment.yml
RUN conda run -n gaussian_splatting pip install submodules/diff-gaussian-rasterization --no-build-isolation
RUN conda run -n gaussian_splatting pip install submodules/simple-knn --no-build-isolation
RUN conda run -n gaussian_splatting pip install submodules/fused-ssim --no-build-isolation

RUN conda env create -f aruco.yaml
RUN conda run -n aruco pip install "git+https://github.com/meyerls/aruco-estimator.git"


RUN conda init bash
ENTRYPOINT ["/bin/bash"]

