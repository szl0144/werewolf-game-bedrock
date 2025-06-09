from openai import OpenAI

def open_ai_whisper(audio_path):
    with OpenAI() as client:
        with open(audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1", 
                language="zh",
                prompt="这里会出现的地点名词有主院、瓷器房、前庭、码头、酒窖、棋房、凉亭、灶房、后花园、正房、茶室、书画房、东厢房",
                file=audio_file,
                response_format="text"
            )

    return transcription