# Speech Emotion Recognition

A speech emotion recognition project using the RAVDESS dataset, MFCC-based acoustic features, and a locally trained PyTorch CNN-BiLSTM model.

## Supported emotions

The model classifies eight emotions:

- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

## Approach

1. Load speech audio at 16 kHz.
2. Extract 13 MFCCs with delta and delta-delta features, producing 39 feature channels.
3. Normalize features using statistics calculated from the training split.
4. Train a CNN-BiLSTM classifier in PyTorch.
5. Evaluate with accuracy, macro F1-score, classification report, and confusion matrix.

The project does not use Hugging Face or pretrained Wav2Vec2 models.

## Dataset

This project uses the [RAVDESS](https://zenodo.org/records/1188976) speech audio dataset. Place the extracted dataset in:

```text
Audio_Speech_Actors_01-24_16k/
```

The expected structure is:

```text
Audio_Speech_Actors_01-24_16k/
  Actor_01/
  Actor_02/
  ...
  Actor_24/
```

The data is split by speaker:

- Training: actors 1-16
- Validation: actors 17-20
- Testing: actors 21-24

The dataset, trained checkpoints, and generated feature arrays are excluded from this repository because of their size. Keep them locally or use Git LFS for private storage.

## Installation

Use Python 3.10 or newer and install the dependencies:

```powershell
python -m pip install numpy pandas librosa scikit-learn torch soundfile
```

## Prepare metadata

```powershell
python .\src\prepare_dataset.py
```

## Build features

The recommended pipeline is version 2, which creates MFCC, delta, and delta-delta features:

```powershell
python .\src\build_features_v2.py
```

This creates the `features_v2/` files used by the v2 model.

## Train the model

```powershell
python .\src\train_model_v2.py
```

The best checkpoint is saved locally as:

```text
models/best_speech_emotion_v2.pth
```

Evaluation results and the confusion matrix are written to `results/`.

## Predict an audio file

Run the prediction script with the audio path as an argument:

```powershell
python .\src\predict.py "C:\path\to\your\audio.wav"
```

Example:

```powershell
python .\src\predict.py ".\Audio_Speech_Actors_01-24_16k\Actor_01\03-01-01-01-01-01-01.wav"
```

The script prints the predicted emotion and probabilities for all eight classes.

## Project structure

```text
src/
  prepare_dataset.py
  build_features_v2.py
  train_model_v2.py
  predict.py
features_v2/       Generated normalized features, kept locally
models/            Trained checkpoints, kept locally
results/           Evaluation outputs
```

## License and dataset note

Check the RAVDESS dataset license and citation requirements before redistributing the dataset or derived files.
