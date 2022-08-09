# MusicVAE를 활용하여 4마디의 드럼 샘플 추출하기

## 0. 개요
- Groove MIDI Dataset을 활용하여 Magenta 라이브러리 내의 MusicVAE 모델을 학습시켜 4마디 분량의 드럼 샘플을 추출하는 과정.
- Google Colab에서 Magenta 라이브러리 내의 코드를 상황에 맞춘 arguments를 입력한 스크립트로 실행하여 전 과정을 수행.

## 1. 전처리
- Groove MIDI Dataset 중 미리 4마디로 분절 처리 된 드럼 midi 데이터셋인 `groove/4bar-midionly`(https://www.tensorflow.org/datasets/catalog/groove?hl=en#groove4bar-midionly)를 학습 데이터로 활용.
- `magenta/models/music_vae/data.py` 내부의 `GrooveConverter` 클래스를 활용, `hits`, `timing offsets`, `velocities`의 정보를 담는 형태로 전처리.

## 2. 학습
- 학습에 쓰인 모델의 설정값은 `magenta/models/music_vae/configs.py` 의 `groovae_4bar`
- 상세 코드는 다음과 같음.
```python
CONFIG_MAP['groovae_4bar'] = Config(
    model=MusicVAE(lstm_models.BidirectionalLstmEncoder(),
                   lstm_models.GrooveLstmDecoder()),
    hparams=merge_hparams(
        lstm_models.get_default_hparams(),
        HParams(
            batch_size=512,
            max_seq_len=16 * 4,  # 4 bars w/ 16 steps per bar
            z_size=256,
            enc_rnn_size=[512],
            dec_rnn_size=[256, 256],
            max_beta=0.2,
            free_bits=48,
            dropout_keep_prob=0.3,
        )),
    note_sequence_augmenter=None,
    data_converter=data.GrooveConverter(
        split_bars=4, steps_per_quarter=4, quarters_per_bar=4,
        max_tensors_per_notesequence=20,
        pitch_classes=data.ROLAND_DRUM_PITCH_CLASSES,
        inference_pitch_classes=data.REDUCED_DRUM_PITCH_CLASSES),
    tfds_name='groove/4bar-midionly',
)
```
- dataset에서 오직 `hits` 부분만 input으로 하여 `BidirectionalLstmEncoder`를 통해 latent vector로 변환됨
- 이 latent vector를 input으로 받은 `GrooveLstmDecoder`는 `hits`뿐만 아니라 정보가 주어지지 않은 `timing offsets`와 `velocities`까지 예측하고, 이 세 요소의 CrossEntropy loss를 최소화 하는 방향으로 학습이 진행 됨.
- 결과적으로 `hits`만으로 `timing offsets`와 `velocities`를 이해시켜 모델로 하여금 드럼의 groove를 이해시키는 것이 목표.

## 3. 생성
- K-Nearest Neighbors 방식으로 4마디의 드럼 연주를 샘플링, 논문 상의 K값은 20.
- 샘플링 된 값을 다시 Midi 포멧으로 변환.
- 본 과제에선 총 3000 step 학습 한 모델을 활용하여 5개의 각기 다른 4마디 드럼 연주를 샘플링 하여 저장함.

## 4. References
- Magenta (https://github.com/magenta/magenta)
- A Hierarchical Latent Vector Model for Learning Long-Term Structure in Music (https://arxiv.org/abs/1803.05428)
- Learning to Groove with Inverse Sequence Transformations (https://arxiv.org/abs/1905.06118)
- Groove MIDI Dataset (https://magenta.tensorflow.org/datasets/groove)