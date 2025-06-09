# Playing Werewolf with Claude 3 Opus

## What is it
This Python-based AI Chatbot Assistant helps players analyze in-game conversations to identify potential werewolves in the popular mobile game "DreamStar." By processing and interpreting player dialogues during Werewolf games, the chatbot utilizes the Claude 3 Opus on Amazon Bedrock to highlight suspicious behaviors and suggest possible werewolves. This tool can be a valuable asset for players looking to enhance their strategic play in Werewolf games.

<div align="center">
  <img src="https://github.com/szl0144/werewolf-game-bedrock/assets/40918217/dace61fc-5093-47f0-a290-747243d9e7f9" alt="Games' conversation Part">
  <p>Games' conversation Part</p>
</div>

## Cloud Architecture 

### Phase 1
![Architecture2](https://github.com/szl0144/werewolf-game-bedrock/assets/40918217/e39df7a9-038b-4928-a06f-2aee09c0e6fb)

### Phase 2
![architecture](https://github.com/szl0144/werewolf-game-bedrock/assets/40918217/b7fe97fc-71e1-494e-adff-dad105ea88ff)

## Installation
FFMPEG is needed as a prerequisite to install the requirements

On Ubuntu or Debian:
sudo apt update && sudo apt install ffmpeg

On Arch Linux:
sudo pacman -S ffmpeg

On macOS using Homebrew:
brew install ffmpeg

On Windows using Chocolatey:
choco install ffmpeg

On Windows using Scoop:
scoop install ffmpeg

### Dependencies
Install the following dependencies:

pip install -r requirements.txt

Requirements include:
1. whisper
2. boto3
3. pyannote.audio
4. setuptools (version 59.5.0)
5. speechbrain (version 0.5.16)

## Prediction Result
Claude Opus successfully predicted the ID of 2 werewolves in the game based on the in-game conversations.

### Claude 3 Opus Output
![Claude Opus Output](https://github.com/szl0144/werewolf-game-bedrock/assets/40918217/5325b737-bfec-4cde-a364-9908bdacb568)

### Real Player Identities
![Player Identities](https://github.com/szl0144/werewolf-game-bedrock/assets/40918217/a5d3fd2e-4941-47da-824d-123ccc2dc53d)