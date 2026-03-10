#! /usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

ARCH=$(uname -m)

echo "Installing essentials"
sudo DEBIAN_FRONTEND=noninteractive apt update
sudo DEBIAN_FRONTEND=noninteractive apt install -y file wget git build-essential gcc g++ gdb cmake make ninja-build curl openssh-client openssh-server

echo "Installing ripgrep and fd and jq"
sudo DEBIAN_FRONTEND=noninteractive apt install -y ripgrep fd-find jq

echo "Installing pipx"
sudo DEBIAN_FRONTEND=noninteractive apt install -y pipx
#echo "Pipx installing vectorcode"
#pipx install vectorcode --python python3.12

echo "Installing neovim"
mkdir -p /home/coder/neovim
if [[ $(uname -m) == "x86_64" ]]; then
    wget -P /home/coder/neovim https://github.com/neovim/neovim/releases/download/v0.11.6/nvim-linux-x86_64.appimage
    cd /home/coder/neovim
    chmod u+x nvim-linux-x86_64.appimage
    ./nvim-linux-x86_64.appimage --appimage-extract
    cd /usr/local/bin
    ln -s /home/coder/neovim/squashfs-root/usr/bin/nvim .
    cd
elif [[ $(uname -m) == "aarch64" ]]; then
    wget -P /home/coder/neovim https://github.com/neovim/neovim/releases/download/v0.11.6/nvim-linux-arm64.appimage
    cd /home/coder/neovim
    chmod u+x nvim-linux-arm64.appimage
    ./nvim-linux-arm64.appimage --appimage-extract
    cd /usr/local/bin
    ln -s /home/coder/neovim/squashfs-root/usr/bin/nvim .
    cd
fi

echo "Installing tmux"
sudo DEBIAN_FRONTEND=noninteractive apt install -y tmux
# TODO: how is this install different from bind mounting .tmux
# Is there a better tmux configuration tool out there? tmux-config is not being maintained
# Rest moved to post-create.sh

echo "Installing clangd"
sudo DEBIAN_FRONTEND=noninteractive apt install -y software-properties-common gnupg lsb-release
LLVM_VERSION=20
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
./llvm.sh ${LLVM_VERSION}
sudo DEBIAN_FRONTEND=noninteractive apt install -y clangd-${LLVM_VERSION}
update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-${LLVM_VERSION} 100
rm llvm.sh

echo "Installing nsight systems"
temp_deb="$(mktemp)" && \
  wget -O "$temp_deb" https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_6/NsightSystems-linux-cli-public-2024.6.1.90-3490548.deb && \
   sudo dpkg -i "$temp_deb" && \
  rm -f "$temp_deb"

echo "Installing bash-completion"
sudo DEBIAN_FRONTEND=noninteractive apt install -y bash-completion

echo "Installing nodejs"
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo DEBIAN_FRONTEND=noninteractive apt install -y nodejs

echo "Installing opencode"
sudo npm install -g opencode-ai

# Clean up apt cache to reduce image size
rm -rf /var/lib/apt/lists/*

# install.sh runs as root during image build but installs into /home/coder/. Those files will be root-owned. 
# So we need explicity chown
chown -R coder:coder /home/coder/
