# Playing Werewolf with Claude 3 Opus
## What is it
This Python-based AI Chatbot Assistant helps players analyze in-game conversations to identify potential werewolves in the popular mobile game "DreamStar." By processing and interpreting player dialogues during Werewolf games, the chatbot utilizes the Claude 3 Opus on Amazon Bedrock to highlight suspicious behaviors and suggest possible werewolves. This tool can be a valuable asset for players looking to enhance their strategic play in Werewolf games.
<div align="center">
  <img src="https://github.com/szl0144/werewolf-game-bedrock/assets/40918217/dace61fc-5093-47f0-a290-747243d9e7f9" alt="Image Description">
  <p style="text-align: center;">Games' conversation Part</p>
</div>

## Cloud Architecture 

### Phase 1
![Architecture2](https://github.com/szl0144/werewolf-game-bedrock/assets/40918217/e39df7a9-038b-4928-a06f-2aee09c0e6fb)


### Phase 2
![architecture](https://github.com/szl0144/werewolf-game-bedrock/assets/40918217/b7fe97fc-71e1-494e-adff-dad105ea88ff)

## Installation
FFMPEG is needed as prerequisites to install the requirements

on Ubuntu or Debian
```
sudo apt update && sudo apt install ffmpeg
```
on Arch Linux
```
sudo pacman -S ffmpeg
```
on MacOS using Homebrew (https://brew.sh/)
```
brew install ffmpeg
```
on Windows using Chocolatey (https://chocolatey.org/)
```
choco install ffmpeg
```
on Windows using Scoop (https://scoop.sh/)
```
scoop install ffmpeg
```
The following dependencies are needed
1. Install whisper.
2. Install boto3.
3. Install pyannote.audio.
4. Downgrade setuptools to 59.5.0
5. Downgrade speechbrain to 0.5.16 

```
pip install -r requirements.txt
```

## Prediction Result
Claude Opus successfully predicted the ID of 2 werewolves in the game based on the in-game conversations.
The Output of the Claude 3 Opus.
![image](https://github.com/szl0144/werewolf-game-bedrock/assets/40918217/5325b737-bfec-4cde-a364-9908bdacb568)

The real identities of each player in the game.
![image](https://github.com/szl0144/werewolf-game-bedrock/assets/40918217/a5d3fd2e-4941-47da-824d-123ccc2dc53d)






