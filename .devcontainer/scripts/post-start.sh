#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

echo "Running post-start commands"

# Git config (idempotent - overwrites)
git config --global user.name "Shruti Shivakumar"
git config --global user.email "shruti.shivakumar@gmail.com"
git config --global user.signingkey /home/coder/.ssh/github-signing.pub
git config --global gpg.format ssh
git config --global gpg.ssh.allowedSignersFile /home/coder/.ssh/allowed_signers
git config --global commit.gpgsign true
git config --global tag.gpgsign true
git config --global merge.gpgsign true
git config --global --add safe.directory /home/coder/cudf

# Bashrc modifications (idempotent - check before append)
grep -q "unset PROMPT_COMMAND" ~/.bashrc || echo "unset PROMPT_COMMAND" >>~/.bashrc
grep -q "XDG_CONFIG_HOME" ~/.bashrc || echo 'export XDG_CONFIG_HOME=/home/coder/.config' >>~/.bashrc

# Bash completion (idempotent)
if ! grep -q "/etc/bash_completion" ~/.bashrc; then
  cat >>~/.bashrc <<'EOF'
if [ -f /etc/bash_completion ]; then
    . /etc/bash_completion
fi
EOF
fi

# Path to the encrypted token files
ENCRYPTED_TOKEN_FILES=("/home/coder/.oneapi-perflab-nvidia.key.gpg" "/home/coder/.inference-nvidia.key.gpg")
TOKEN_ENV_VAR_NAMES=("AZURE_OPENAI_API_KEY" "SHRSHI_NVINFERENCE_APIKEY")

# Check if any tokens need decryption
needs_decryption=false
for i in "${!ENCRYPTED_TOKEN_FILES[@]}"; do
  if [ -f "${ENCRYPTED_TOKEN_FILES[$i]}" ] && ! grep -q "export ${TOKEN_ENV_VAR_NAMES[$i]}" ~/.bashrc; then
    needs_decryption=true
    break
  fi
done

# Prompt for passphrase once if needed
if [ "$needs_decryption" = true ]; then
  read -s -p "Enter GPG passphrase for API keys: " GPG_PASSPHRASE
  echo
fi

# Function to decrypt token and add to bashrc
# Returns 0 on success, 1 on failure
setup_token() {
  ENCRYPTED_TOKEN_FILE=$1
  TOKEN_ENV_VAR_NAME=$2
  if [ -f "$ENCRYPTED_TOKEN_FILE" ]; then
    # Skip if already in bashrc
    if grep -q "export $TOKEN_ENV_VAR_NAME" ~/.bashrc; then
      echo "Auth token $TOKEN_ENV_VAR_NAME already in .bashrc"
      return 0
    fi

    echo "Decrypting $TOKEN_ENV_VAR_NAME..."
    TOKEN=$(gpg --quiet --batch --passphrase "$GPG_PASSPHRASE" --decrypt "$ENCRYPTED_TOKEN_FILE" 2>/dev/null)

    if [ $? -eq 0 ] && [ -n "$TOKEN" ]; then
      echo "# Auth token from GPG" >>~/.bashrc
      echo "export $TOKEN_ENV_VAR_NAME=\"$TOKEN\"" >>~/.bashrc
      echo "Auth token $TOKEN_ENV_VAR_NAME added to environment variables."
      return 0
    else
      echo "Failed to decrypt $TOKEN_ENV_VAR_NAME. Aborting token setup."
      return 1
    fi
  else
    echo "Warning: Auth token file not found at $ENCRYPTED_TOKEN_FILE"
    return 0
  fi
}

# Configure keys for API access
for i in "${!ENCRYPTED_TOKEN_FILES[@]}"; do
  if ! setup_token "${ENCRYPTED_TOKEN_FILES[$i]}" "${TOKEN_ENV_VAR_NAMES[$i]}"; then
    unset GPG_PASSPHRASE
    break
  fi
done

# Clear passphrase from memory
unset GPG_PASSPHRASE

# Tmux config (skip if already exists)
if [ ! -d /home/coder/tmux-config ]; then
  echo "Installing tmux config"
  git clone https://github.com/samoshkin/tmux-config.git /home/coder/tmux-config
  tee /home/coder/tmux-config/tmux/tmux-conf.patch <<'EOF'
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
  /home/coder/tmux-config/install.sh
  # Enable clipboard
  sed -i '/^set -g mouse on/a set -g set-clipboard on' /home/coder/tmux-config/tmux/tmux.conf
  sed -i 's/^set -g default-terminal.*/set -g default-terminal "tmux-256color"/' /home/coder/tmux-config/tmux/tmux.conf
  sed -i '/^set -g default-terminal/a set -ga terminal-overrides ",xterm-256color:Tc,tmux-256color:Tc"' /home/coder/tmux-config/tmux/tmux.conf
  sed -i '/^set -ga terminal-overrides/a set -gq utf8 on' /home/coder/tmux-config/tmux/tmux.conf
  sed -i '/^set -gq utf8 on/a set -gq status-utf8 on' /home/coder/tmux-config/tmux/tmux.conf
  ln -sf ~/tmux-config/tmux/tmux.conf ~/.tmux.conf
else
  echo "tmux-config already exists, skipping"
fi

echo "Post-start complete"
