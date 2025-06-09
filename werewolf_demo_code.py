import base64
import json
import boto3
import whisper
import io
from pydub import AudioSegment
from pydub.playback import play
from pyannote.audio import Pipeline
from pyannote_whisper.utils import diarize_text
import os

class AudioProcessor:
    def __init__(self):
        self.pipeline = None
        self.model = None
        self.bedrock_client = None
        self.polly_client = None
    
    def _get_pipeline(self):
        if self.pipeline is None:
            auth_token = os.getenv("HUGGINGFACE_TOKEN")
            if not auth_token:
                raise ValueError("HUGGINGFACE_TOKEN environment variable not set")
            self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=auth_token)
        return self.pipeline
    
    def _get_whisper_model(self):
        if self.model is None:
            self.model = whisper.load_model("tiny")
        return self.model
    
    def _get_bedrock_client(self):
        if self.bedrock_client is None:
            self.bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="us-west-2")
        return self.bedrock_client
    
    def _get_polly_client(self):
        if self.polly_client is None:
            self.polly_client = boto3.client('polly')
        return self.polly_client

    def pyannote_whisper(self, audio_path="./Audio.wav"):
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        whisper_output_lines = []
        pipeline = self._get_pipeline()
        model = self._get_whisper_model()
        
        prompt = "这里会出现的地点名词有:主院、瓷器房、前庭、码头、酒窖、棋房、凉亭、灶房、后花园、正房、茶室、书画房、东厢房"
        asr_result = model.transcribe(audio_path, prompt=prompt, language="zh")
        diarization_result = pipeline(audio_path, num_speakers=8)
        final_result = diarize_text(asr_result, diarization_result)

        for seg, spk, sent in final_result:
            line = f'{seg.start:.2f} {seg.end:.2f} {spk} {sent}'
            whisper_output_lines.append(line)
        
        return "\n".join(whisper_output_lines)

    def bedrock_invoke(self, base64_string, prompt):
        bedrock = self._get_bedrock_client()
        
        game_instructions = """我在玩元梦之星狼人杀游戏。游戏共有10人，有6名平民，2名狼人，2名中立阵营。每个玩家都有自己的职业身份。你是平民阵营，你需要通过10名玩家的对话找到不合理的地方，猜出来谁是狼人。我上传的图片就是这个游戏里的地图位置。狼人可以使用密道在地图中相互穿梭，直接到达其他的密道口，前庭的密道通往东厢房，后花园的密道连接瓷器房、后花园、正方密道通往棋房和灶房，图片中蓝色的就是密道口。

平民可以做任务，正常匹配对局，平民一个任务2%进度，学者4%，任务做完50个获胜，任务点为（1）主院：摘莲花、摘果子 （2）瓷器房：擦拭瓷器 （3）前庭：浇花1/2 （4）码头：取货、搬货1/2 （5）酒窖：寻找证物、摆放酒坛 6）棋房：对弈、整理棋盘 （7）凉亭：钓鱼 （8）灶房：劈柴、煮鱼、搬货2/2  （9）后花园：寻找玉佩、浇花2/2  （10）正房：焚香、砸锁 （11）茶室：验毒 （12）书画房：鉴别仿品 （13） 东厢房（小院） ：擦拭院墙、清扫落叶 

规则特性：

 1.  每个角色仅可以主动拉一次主院铃。可以拉多个人的尸体铃。炸弹炸死，没有尸体，无法拉尸体铃。

 2.  游戏有总时长限制，总时间为30分钟。超时狼人获胜。

 3.  紧急任务-灭火。90秒内必须灭火（地点：灶房，码头），超时未完成任务，狼人获胜。

 4.  紧急任务-驱散迷雾。限制视野，无时间限制。地点：后花园水井。

 5.  紧急任务-吃饭。地点：灶房。

 6.  紧急任务-起雾冷却时间为60秒。紧急任务-吃饭冷却时间为180秒。每次会议后，紧急任务冷却时间重置为60秒。

 7.  尸体。狼人尸体仍可以发布紧急任务。平民尸体仍可以做任务，学者做任务额外增加任务进度能力仍在，平民也可以继续做任务（也增加进度）。

8. 狼人与关门的技能，把人关到封闭房间里不能离开。


该游戏有以下角色
一、平民阵营

1、平民：常规的身份，可以进行任务、报告尸体、没有特殊能量;

2、警长：能够攻击其他玩家，如果是平民阵营就可以存活，警长自己淘汰;

3、天使：保护一名玩家，在本次回合中可以免疫一次攻击，在每个回合只能使用一次;

4、侦探：可以调查一名玩家，知晓它所在的阵营

5、主持人：投票的时候，你的投票数量被计作为两次;

6、勇者：完成四个普通任务之后，获得武器有攻击能力，每一局只能使用一次;

7、学者：完成任务的时候可以使用总任务的进度增加很多。

二、狼人阵营

1、狼人：常规身份可以攻击平民、使用密道、没有很多特殊的能力;

2、伪装狼：可以随机变身成为一个玩家持续一段时间;

3、炸弹狼：将一枚炸弹放在玩家身上，会自动爆炸淘汰携带者;

4、刺客狼：可以在会议中猜测一名玩家的身份，猜对就可以暗算掉玩家，猜错就会淘汰;

三、中立阵营

1、小丑：在会议期间是会被投票驱逐，获得个人胜利;
2、臭鼬：可以用臭味标记玩家，在一个回合之内可以存活玩家被标记，获得胜利。
3、赌徒：会议期间赌徒可以猜测一名玩家，猜对三名就可以获得胜利;
4、赏金猎人：每个回合有一个赏金目标，亲手攻击淘汰三名目标获得胜利;


你需要根据我下面输入的10个对话，先找出2个狼人，报出狼人序号。

这局所有人身份共有10个，侦探，哨子、消防员、主持人、学者、勇士、大力狼、伪装狼、小丑、赌徒
狼有技能关门，把人关到房间里。
这局8号和10号被淘汰了，你是7号消防员身份，有2个狼，帮我找出来一个并分析，并猜测每个人的身份。下面是按顺序发言:"""
        
        prompt_config = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_string,
                            },
                        },
                        {"type": "text", "text": game_instructions + prompt},
                    ],
                }
            ],
        }
        
        try:
            body = json.dumps(prompt_config)
            response = bedrock.invoke_model(
                contentType="application/json", 
                body=body, 
                modelId="anthropic.claude-3-opus-20240229-v1:0"
            )
            response_body = json.loads(response.get("body").read())
            return response_body.get("content")[0].get("text")
        except Exception as e:
            raise Exception(f"Bedrock API call failed: {str(e)}")

    def aws_polly_text_to_speech(self, text):
        client = self._get_polly_client()
        
        try:
            response = client.synthesize_speech(
                Text=text,
                LanguageCode="cmn-CN",
                OutputFormat='mp3',
                VoiceId='Zhiyu'
            )
            return response['AudioStream'].read()
        except Exception as e:
            raise Exception(f"Polly TTS failed: {str(e)}")

def main():
    processor = AudioProcessor()
    
    try:
        prompt = processor.pyannote_whisper()
        
        image_path = 'Map.png'
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
            base64_string = base64.b64encode(image_data).decode('utf-8')
        
        results = processor.bedrock_invoke(base64_string, prompt)
        stream_base64 = processor.aws_polly_text_to_speech(results)
        audio = AudioSegment.from_file(io.BytesIO(stream_base64), format="mp3")
        play(audio)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()