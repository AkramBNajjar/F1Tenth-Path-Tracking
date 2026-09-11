# ROS 2 Jazzy on Ubuntu 24.04. Official images publish arm64, so this builds
# natively on Apple Silicon with no emulation.
FROM ros:jazzy-ros-base

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-pip git ros-jazzy-ackermann-msgs \
    && rm -rf /var/lib/apt/lists/*

# f1tenth_gym pins are stale (see SETUP_MACOS.md). Install without them.
RUN pip3 install --break-system-packages --no-cache-dir \
      "numpy<2" scipy numba pillow pyyaml "pyglet==2.0.15" "gym==0.26.2"

RUN git clone --depth 1 https://github.com/f1tenth/f1tenth_gym.git /opt/f1tenth_gym \
    && pip3 install --break-system-packages --no-deps -e /opt/f1tenth_gym

ENV F1TENTH_EXAMPLES=/opt/f1tenth_gym/examples
WORKDIR /ws
CMD ["bash"]
