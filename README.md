# "Hey Daisy": Voice Command Interface for Spot🌼

**Hey Daisy** is a voice-controlled command interface for a Boston Dynamics Spot robot. It uses speech recognition (via the [Vosk](https://alphacephei.com/vosk/) model) to capture and parse spoken commands, allowing a user to direct the robot to perform basic actions like sitting, standing, stopping, or rotating—all through natural language.

---
## Features

- 🎤 **Voice Command Recognition**: Uses [Vosk](https://alphacephei.com/vosk/) for speech-to-text processing
- 🤖 **Robot Command Dispatch**: Converts recognized text into robot actions
- 🌐 **Multilingual Support**: Currently supports English (`en-us`) and Russian (`ru`)
- 🧠 **Human-in-the-Loop**: Requires user confirmation before executing any command

---

## Requirements

- Python 3.10.12+
- A working microphone
- [Vosk](https://alphacephei.com/vosk/) model(s) downloaded locally
- Boston Dynamics Spot robot
### Python Packages

Install dependencies:

```bash
pip install sounddevice vosk opencv-python numpy scipy bosdyn-client
```

---
### Options

- `--list-devices` or `-l`: Print available audio input devices and exit
- `--device` or `-d`: Input device index or substring
- `--model` or `-m`: Specifies language of the model: `en-us` *(default)* or `ru`
- `--help` or `-h`: Shows usage summary

**To run:**
```bash
python3 hey_daisy.py -d <mic_id> -m <language_code>
```

---

## Example Interaction
1. Speak a command like "stand up" or "повернись направо"
2. Daisy transcribes the audio and prints the interpreted instruction
3. You'll be prompted:  
   `Received instruction to stand. Do you approve execution? (y/[N])`
4. `y` to approve, `n` or Enter to prevent execution
---
## Command Parsing Configuration
The `parse_lists.json` file defines how spoken phrases are interpreted as robot commands in different languages. It maps each command to a list of possible trigger phrases that may be recognized from speech input.

**For example:**

```json
{
  "en-us": {
    "sit": ["sit", "sit down"],
    "stand": ["stand", "stand up"],
    "stop": ["stop", "stops"],
    "rotate clockwise": ["rotate right", "rotate clockwise"],
    "rotate counterclockwise": ["rotate left", "rotate counterclockwise"],
    "turn around": ["turn around"]
  }
}
```

## Supported Commands

| Command                   | English Keywords                         | Russian Keywords                              |
| ------------------------- | ---------------------------------------- | --------------------------------------------- |
| `sit`                     | "sit", "sit down", "sitting"             | "сядь", "сидеть", "присядь"                   |
| `stand`                   | "stand", "stand up"                      | "встать", "встань", "вставай"                 |
| `stop`                    | "stop", "stopping"                       | "стой", "остановись"                          |
| `rotate clockwise`        | "rotate right", "rotate clockwise"       | "повернись вправо", "повернись по часовой"    |
| `rotate counterclockwise` | "rotate left", "rotate counterclockwise" | "повернись влево", "повернись против часовой" |
| `turn around`             | "turn around"                            | "развернись", "развернуться"                  |

