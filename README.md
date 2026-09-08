# Speech Emotion Recognition

A speech emotion recognition project using the RAVDESS dataset, MFCC-based acoustic features, and a locally trained PyTorch CNN-BiLSTM model.

## Web Application (Live Demo)
A user-friendly web interface is available to test the model! 
You can run it locally using Streamlit, or deploy it directly via Streamlit Community Cloud. 

To run the web UI locally:
```powershell
pip install -r requirements.txt
streamlit run app.py
```
This will open a browser window where you can drag and drop an audio file (.wav, .mp3) and instantly see the predicted emotion and confidence score.

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

The dataset is split by speaker:

```text
Audio_Speech_Actors_01-24_16k/
  Actor_01/
  ...
  Actor_24/
```

- Training: actors 1-16
- Validation: actors 17-20
- Testing: actors 21-24

The dataset, trained checkpoints, and generated feature arrays are excluded from this repository because of their size. Keep them locally or use Git LFS for private storage.

## Prepare metadata

```powershell
python .\src\prepare_dataset.py
```

## Build features

The recommended pipeline is V3D, which creates 45-channel acoustic features:

```powershell
python .\src\build_feaatures_v3c.py
```

This creates the `features_v3c/` files used by the V3D model.

## Train the model

```powershell
python .\src\train_model_main.py
```

The best checkpoint is saved locally as:

```text
models/main_speech_emotion_model.pth
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
  build_feaatures_v3c.py
  train_model_main.py
  predict.py
features_v3c/       Generated normalized features, kept locally
models/            Trained checkpoints, kept locally
results/           Evaluation outputs
```

## License and dataset note

Check the RAVDESS dataset license and citation requirements before redistributing the dataset or derived files.
