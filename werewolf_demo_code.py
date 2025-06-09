import base64
import json
import boto3
import whisper
import io
from pydub import AudioSegment
from pydub.playback import play
from pyannote.audio import Pipeline
from pyannote_whisper.utils import diarize_text
import concurrent.futures
import os

def pyannote_whisper():
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=os.getenv('PYANNOTE_TOKEN'))
    model = whisper.load_model("tiny")
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        transcribe_future = executor.submit(model.transcribe, "./Audio.wav", prompt="这里会出现的地点名词有:主院、瓷器房、前庭、码头、酒窖、棋房、凉亭、灶房、后花园、正房、茶室、书画房、东厢房", language="zh")
        diarization_future = executor.submit(pipeline, "./Audio.wav", num_speakers=8)
        
        asr_result = transcribe_future.result()
        diarization_result = diarization_future.result()
    
    final_result = diarize_text(asr_result, diarization_result)
    return "\n".join(f'{seg.start:.2f} {seg.end:.2f} {spk} {sent}' for seg, spk, sent in final_result)

def bedrock_invoke(base64_string, prompt):
    bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-west-2")
    
    prompt_config = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64_string}},
                {"type": "text", "text": prompt}
            ]
        }]
    }

    try:
        response = bedrock.invoke_model(
            contentType="application/json", 
            body=json.dumps(prompt_config), 
            modelId="anthropic.claude-3-opus-20240229-v1:0"
        )
        response_body = json.loads(response.get("body").read())
        return response_body.get("content")[0].get("text")
    except Exception as e:
        print(f"Error in Bedrock invoke: {e}")
        return ""

def aws_polly_text_to_speech(text):
    client = boto3.client('polly')
    
    try:
        response = client.synthesize_speech(
            Text=text,
            LanguageCode="cmn-CN",
            OutputFormat='mp3',
            VoiceId='Zhiyu'
        )
        return response['AudioStream'].read()
    except Exception as e:
        print(f"Text-to-speech error: {e}")
        return None

def main():
    try:
        prompt = pyannote_whisper()
        
        with open('Map.png', 'rb') as image_file:
            base64_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        results = bedrock_invoke(base64_string, prompt)
        
        if results:
            stream_base64 = aws_polly_text_to_speech(results)
            
            if stream_base64:
                audio = AudioSegment.from_file(io.BytesIO(stream_base64), format="mp3")
                play(audio)
    except Exception as e:
        print(f"Main process error: {e}")

if __name__ == "__main__":
    main()