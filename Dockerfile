# Builds axon-server and axon on Debian with clang and copies the two binaries into an image
# that holds nothing else but the C and C++ runtime. The model is mounted at run time:
#
#   docker build -t axon-infer .
#   docker run --rm -p 8080:8080 -v "$PWD/models:/models:ro" axon-infer -m /models/SmolLM2-135M-Instruct-f16.gguf
#
# On an arm64 host the build gets the NEON kernel table, on x86 the scalar reference table.

FROM debian:trixie-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
      clang cmake ninja-build git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
      -DAXON_BUILD_TESTS=OFF -DAXON_BUILD_BENCH=OFF \
    && cmake --build build --target axon-server axon

FROM debian:trixie-slim
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --no-create-home axon
COPY --from=build /src/build/apps/axon-server/axon-server /usr/local/bin/axon-server
COPY --from=build /src/build/apps/axon/axon /usr/local/bin/axon
USER axon
VOLUME ["/models"]
EXPOSE 8080
ENTRYPOINT ["axon-server", "--host", "0.0.0.0", "--port", "8080"]
CMD ["--help"]
