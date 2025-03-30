import os
import re

from multiprocessing import freeze_support
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

root_path = "../audio"
name = 'wavs'
name_folder_path = os.path.join(root_path, name)
metadata = 'metadata.txt'

def formatter(name_folder_path, metadata, **kwargs):
    regex = r'([+#"]+)|(<.*?>+)|{.*?}+'
    file = os.path.join(name_folder_path, metadata)
    item_list = []
    speaker = name
    
    with open(file, 'r', encoding='utf-8') as tf:
         for line in tf:
             colmn = line.split('|')
             audio_file = f"{name_folder_path}/{colmn[0]}.wav"
             text = re.sub(regex, "", colmn[1])
             
             #print(f"Original: {colmn[1]} | Formatted: {text}")
             
             item_list.append({
                 "text": text,
                 "audio_file": audio_file,
                 "speaker_name": speaker,
                 "root_path": name_folder_path
             })
    
    return item_list

if __name__ == '__main__':
    freeze_support()
    
    dataset_config = BaseDatasetConfig(
        formatter=None,
        meta_file_train="metadata.txt",
        language="en-us",
        path=name_folder_path
    )

    audio_config = VitsAudioConfig(
        sample_rate=22050,
        fft_size=1024,
        win_length=1024,
        hop_length=256,
        num_mels=80,
        mel_fmin=0,
        mel_fmax=None
    )

    config = VitsConfig(
        audio=audio_config,
        run_name="vits_custom",
        batch_size=4,
        eval_batch_size=8,
        num_loader_workers=4,   
        num_eval_loader_workers=2,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=500,
        text_cleaner="english_cleaners",
        use_phonemes=True,
        eval_split_size=0.15,
        lr_gen=3e-5,
        lr_disc=3e-5,
        phoneme_language="en-us",
        phoneme_cache_path=os.path.join(name_folder_path, "phoneme_cache"),
        compute_input_seq_cache=True,
        compute_f0=True,
        print_step=25,
        print_eval=True,
        save_best_after=100,
        save_checkpoints=True,
        save_all_best=True,
        mixed_precision=False,
        datasets=[dataset_config],
        cudnn_benchmark=False,
            lr_scheduler_gen="CyclicLR",
            lr_scheduler_gen_params={"base_lr": 3e-5, "max_lr": 1e-4, "step_size_up": 200, "mode": "triangular"},
        lr_scheduler_disc="CyclicLR",
        lr_scheduler_disc_params={"base_lr": 3e-5, "max_lr": 1e-4, "step_size_up": 200, "mode": "triangular"}
    )
    
    audio_processor = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)
    
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_size=config.eval_split_size,
        eval_split_max_size=config.eval_split_max_size,
        formatter=formatter
    )
    
    print(f"Train Samples: {len(train_samples)}")
    print(f"Eval Samples: {len(eval_samples)}")
    
    model = Vits(config, audio_processor, tokenizer, speaker_manager=None)
    model.text_encoder.requires_grad_ = False
    model.duration_predictor.requires_grad_ = True
    model.posterior_encoder.requires_grad_ = True
    
    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path="./",
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    
    trainer.fit()