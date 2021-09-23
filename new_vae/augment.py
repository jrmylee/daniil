from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
# import torch
# from torch_audiomentations import Compose, AddColoredNoise, Gain, LowPassFilter, HighPassFilter

def augment_audio(audio, midi_filepath):
    sr = 22050
    augment = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.15, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
    ])
    return augment(samples=audio, sample_rate=sr), midi_filepath

# def augment_audio(X_tr, sr):
#     X_torch_tr = torch.tensor(X_tr).reshape((X_tr.shape[0], 1, X_tr.shape[1]))
#     apply_augmentation = Compose(
#         transforms=[
#             LowPassFilter(p=.3),
#             HighPassFilter(p=.3),
#             Gain(p=.5),
#             AddColoredNoise(p=.9),
#             LowPassFilter(p=.3),
#             HighPassFilter(p=.3),
#         ]
#     )
#     torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     X_torch_tr.to(torch_device)
#     return apply_augmentation(X_torch_tr, sample_rate=sr)
