#! /usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

echo "Copying post create script"
# Copy script to a standard location
# Check if source file exists before copying
if [ -f post-create.sh ]; then
    cp post-create.sh /usr/local/bin/local-feature-utils-post-create
    chmod +x /usr/local/bin/local-feature-utils-post-create
    echo "Successfully copied post-create.sh"
else
    echo "Warning: post-create.sh not found"
    exit 1
fi

if [ -f /usr/local/bin/local-feature-utils-post-create ]; then
    echo "post-create.sh copied successfully"
else
    echo "Warning: cannot find the script /usr/local/bin/local-feature-utils-post-create"
    exit 1
fi

ARCH=$(uname -m)

echo "Installing essentials"
DEBIAN_FRONTEND=noninteractive apt update
DEBIAN_FRONTEND=noninteractive apt install -y file wget git build-essential gcc g++ gdb cmake make ninja-build curl openssh-client openssh-server

echo "Installing ripgrep and fd and jq"
DEBIAN_FRONTEND=noninteractive apt install -y ripgrep fd-find jq

echo "Installing pipx"
DEBIAN_FRONTEND=noninteractive apt install -y pipx
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
DEBIAN_FRONTEND=noninteractive apt install -y tmux
echo "unset TMUX" >> /home/coder/.bashrc
git clone https://github.com/samoshkin/tmux-config.git /home/coder/tmux-config
tee /home/coder/tmux-config/tmux/tmux-conf.patch <<EOF
18c18
< set -g prefix C-a
---
> set -g prefix M-q
41,42c41,42
< unbind }    # swap-pane -D
< unbind {    # swap-pane -U
---
> unbind \}    # swap-pane -D
> unbind \{    # swap-pane -U
98c98
< bind \ if '[ #{pane_index} -eq 1 ]' \
---
> bind \\ if '[ #{pane_index} -eq 1 ]' \
EOF
patch /home/coder/tmux-config/tmux/tmux.conf /home/coder/tmux-config/tmux/tmux-conf.patch
./home/coder/tmux-config/install.sh

echo "Installing clangd"
DEBIAN_FRONTEND=noninteractive apt install -y clangd-12
update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-12 100

echo "Installing nsight systems"
temp_deb="(mktemp)" && \
  wget -O "$temp_deb" https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_6/NsightSystems-linux-cli-public-2024.6.1.90-3490548.deb && \
   dpkg -i "$temp_deb" && \
  rm -f "$temp_deb"

echo "Installing packages for claude code sandbox"
DEBIAN_FRONTEND=noninteractive apt install -y bubblewrap
DEBIAN_FRONTEND=noninteractive apt install -y socat

echo "Bash auto completion"
DEBIAN_FRONTEND=noninteractive apt install -y bash-completion
