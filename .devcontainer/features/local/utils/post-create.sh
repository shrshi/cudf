#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


echo "Running post attach commands"

git config --global user.name \"Shruti Shivakumar\"
git config --global user.email \"shruti.shivakumar@gmail.com\"
git config --global user.signingkey /home/coder/.ssh/github-signing.pub
git config --global gpg.format ssh
git config --global gpg.ssh.allowedSignersFile /home/coder/.ssh/allowed_signers
git config --global commit.gpgsign true
git config --global tag.gpgsign true
git config --global merge.gpgsign true

echo "unset PROMPT_COMMAND" >>~/.bashrc

# Path to the encrypted token file
ENCRYPTED_TOKEN_FILE="/home/coder/.oneapi-perflab-nvidia.key.gpg"
# TOKEN_ENV_VAR_NAMES=("PERFLAB_LLM_APIKEY" "ANTHROPIC_AUTH_TOKEN" "AZURE_OPENAI_API_KEY")
TOKEN_ENV_VAR_NAMES=("AZURE_OPENAI_API_KEY")

# Function to decrypt token and add to bashrc
setup_token() {
  if [ -f "$ENCRYPTED_TOKEN_FILE" ]; then
    echo "Decrypting auth token..."

    # Prompt for password to decrypt
    TOKEN=$(gpg --quiet --batch --decrypt "$ENCRYPTED_TOKEN_FILE" 2>/dev/null)

    if [ $? -eq 0 ] && [ ! -z "$TOKEN" ]; then
      for TOKEN_ENV_VAR_NAME in "${TOKEN_ENV_VAR_NAMES[@]}"; do
        # Add to .bashrc if not already there
        if ! grep -q "export $TOKEN_ENV_VAR_NAME" ~/.bashrc; then
          echo "# Auth token from GPG" >>~/.bashrc
          echo "export $TOKEN_ENV_VAR_NAME=\"$TOKEN\"" >>~/.bashrc
          echo "Auth token added to environment variables."
        else
          echo "Auth token already in .bashrc"
        fi
      done
    else
      echo "Failed to decrypt token. Please check your GPG password."
    fi
  else
    echo "Warning: Auth token file not found at $ENCRYPTED_TOKEN_FILE"
  fi
}

# Configure key for OneAPI LLM models to be used by Claude code and CodeCompanion
setup_token

# Configure Claude Code with both Claude and OpenAI models
# Set Base URL to OneAPI Anthropic API Endpoint
# echo 'export ANTHROPIC_BASE_URL=https://llm-proxy.perflab.nvidia.com/anthropic' >>~/.bashrc
# Set the model you want to use
# echo 'export ANTHROPIC_MODEL=claude-3-7-sonnet-20250219' >>~/.bashrc
# Set the small model for background tasks
# echo 'export ANTHROPIC_SMALL_FAST_MODEL=model-router' >>~/.bashrc

# Neovim config home
echo 'export XDG_CONFIG_HOME=/home/coder/.config' >>~/.bashrc

# Github SSH key
touch ~/.ssh/config
cat >~/.ssh/config <<EOF
Host github.com
HostName github.com
User git
IdentityFile ~/.ssh/github
IdentitiesOnly yes
EOF

echo "Installing claude code"
curl -fsSL https://claude.ai/install.sh | bash
echo "Installing goose"
curl -fsSL https://github.com/block/goose/releases/download/stable/download_cli.sh | CONFIGURE=false bash

