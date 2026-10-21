import io
import wave
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from x5a_web_agent.asr import SenseVoiceSpeechToText, WhisperSpeechToText, create_asr
from x5a_web_agent.wrist_commands import parse_wrist_command


def audio(seconds=1.,sample_rate=44100,channels=2):
    t=np.arange(round(seconds*sample_rate))/sample_rate
    pcm=(.2*np.sin(2*np.pi*440*t)*32767).astype('<i2')
    data=io.BytesIO()
    with wave.open(data,'wb') as wav:
        wav.setnchannels(channels);wav.setsampwidth(2);wav.setframerate(sample_rate)
        wav.writeframes(np.repeat(pcm[:,None],channels,axis=1).tobytes())
    return data.getvalue()


def recognizer(asr):
    stream=SimpleNamespace(result=SimpleNamespace(text='<|zh|><|NEUTRAL|><|Speech|>准备。'),accept_waveform=Mock())
    asr._model=SimpleNamespace(create_stream=Mock(return_value=stream),decode_stream=Mock())
    return stream


def test_android_stereo_wav_is_resampled_and_tags_do_not_enter_commands():
    asr=SenseVoiceSpeechToText()
    stream=recognizer(asr)
    text=asr.transcribe(audio(),mime_type='audio/wav')
    rate,samples=stream.accept_waveform.call_args.args
    assert rate==16000 and samples.ndim==1 and len(samples)==16000
    assert .15 < np.max(np.abs(samples)) < .4
    assert text=='准备。' and parse_wrist_command(text)=={'action':'prepare'}


def test_corrupted_audio_cannot_produce_a_robot_command():
    asr=SenseVoiceSpeechToText();recognizer(asr)
    with pytest.raises(RuntimeError):asr.transcribe(b'not audio')
    asr._model.decode_stream.assert_not_called()


def test_overlong_audio_is_rejected_instead_of_truncating_to_a_command():
    asr=SenseVoiceSpeechToText();recognizer(asr)
    with pytest.raises(ValueError,match='30 秒'):asr.transcribe(audio(seconds=30.1,sample_rate=16000,channels=1),'audio/wav')
    asr._model.decode_stream.assert_not_called()


def test_silence_does_not_decode_into_a_hallucinated_command():
    asr=SenseVoiceSpeechToText();recognizer(asr)
    data=io.BytesIO()
    with wave.open(data,'wb') as wav:
        wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(16000);wav.writeframes(b'\0'*32000)
    assert asr.transcribe(data.getvalue(),'audio/wav')==''
    asr._model.decode_stream.assert_not_called()


def test_local_model_is_required_and_old_backend_remains_selectable(tmp_path):
    asr=SenseVoiceSpeechToText(model_dir=tmp_path)
    with pytest.raises(RuntimeError,match='模型未安装'):asr.preload()
    assert not asr.ready
    assert isinstance(create_asr(),SenseVoiceSpeechToText)
    assert isinstance(create_asr('whisper'),WhisperSpeechToText)


@pytest.mark.parametrize('text',['回零位','回到零位','归零位'])
def test_zero_pose_voice_aliases(text):
    assert parse_wrist_command(text)=={'action':'go_home'}
