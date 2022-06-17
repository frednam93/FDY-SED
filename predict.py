from collections import defaultdict

import librosa
from matplotlib import pyplot as plt
from utils.settings import *

from cog import BaseModel, BasePredictor, Input, Path


class Output(BaseModel):
    plot: Path = None
    Json: str = None


class Predictor(BasePredictor):
    def setup(self):
        # Set cuda
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(self.device)

        # load configs
        config = yaml.safe_load(open("./configs/config.yaml", "r"))
        self.sample_rate = config['feature']['sample_rate']
        self.median_window = config['training']["median_window"]
        self.labels = get_labeldict()

        # Load preprocess modules
        self.encoder = get_encoder(self.labels, config['feature'], config['feature']['audio_max_len'])
        self.mel_extractor = setmelspectrogram(config['feature'])
        self.scaler = get_scaler(config["scaler"])

        # Load detection netowrk
        self.net = CRNN(**config["CRNN"])
        self.net.load_state_dict(torch.load('exps/new_exp_gpu=0/best_student.pt'))
        self.net.to(self.device)
        self.net.eval()

    def predict(self,
                input_audio: Path = Input(description="Autio file"),
                output_format: str = Input(description="Recieve outputs as Json or a plot", default="Plot", choices=['Json', 'Plot']),
                threshold: float = Input(description="Detecion probability threshold (relevant to json outputs only)", default=0.5)
                ) -> Output:

        # Load wave form and convert to logmel spectogram
        wav = load_wave_form(str(input_audio), self.encoder.sr)
        wav = wav.unsqueeze(0)
        logmels = self.scaler(take_log(self.mel_extractor(wav)))

        # Network inference
        with torch.no_grad():
            preds, weak_preds = self.net(logmels.to(self.device))

        if output_format == "Json":
            # Decode detection with threshold into events
            json = decode_predictions(self.encoder, preds, threshold=float(threshold), weak_preds=weak_preds, median_window=self.median_window)
            return Output(Json=str(json))
        else:
            # Plot output detection probabilities
            output_path = 'output.png'
            plot(wav[0].cpu().numpy(), preds[0].cpu().numpy(), self.labels, self.sample_rate, output_path)
            return Output(plot=Path(output_path))


def load_wave_form(filepath, target_sample_rate):
    """ Load and resample signal """
    # wav, sr = sf.read(filepath)
    wav, sr = librosa.load(filepath, sr=None)
    wav = to_mono(wav)

    wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sample_rate)

    wav = torch.from_numpy(wav).float()
    wav = normalize_wav(wav)
    return wav


def decode_predictions(encoder, preds, weak_preds=None, threshold=0.5, median_window=None):
    """ Decode detection probabilities into discrete events with a probability threshold"""
    output = preds[0].transpose(0, 1).detach().cpu().numpy() # output size = [frames, n_class]
    if weak_preds is not None:
        for class_idx in range(weak_preds.size(1)):
            if weak_preds[:, class_idx] < threshold:
                output[:, class_idx] = 0
    output = output > threshold
    for mf_idx in range(len(median_window)):
        output[:, mf_idx] = scipy.ndimage.filters.median_filter(output[:, mf_idx], (median_window[mf_idx]))
    events = encoder.decode_strong(output)

    # to Json
    json_dict = defaultdict(lambda : [])
    for (event, start, end) in events:
        json_dict[event].append({"start": start, "end": end})

    return dict(json_dict)

def plot(wave_form, predictions, labels, sample_rate, output_path):
    """Plot signal and detection probabilities"""
    plt.figure(figsize=(15, 10))
    plt.subplot(211)
    N = len(wave_form)
    plt.plot(range(N), wave_form)
    yticks = np.linspace(0, N-1, 10, dtype=int)
    plt.xticks(yticks, [f'{x/sample_rate:.1f}' for x in yticks])
    plt.xlim(0, N)
    plt.ylabel('waveform spec')

    plt.subplot(212)
    plt.title('predictions')
    plt.imshow(predictions, aspect='auto')

    n_frames = predictions.shape[1]
    yticks = np.linspace(0, n_frames-1, 10, dtype=int)
    plt.xticks(yticks, [f'{x/n_frames*10:.1f}' for x in yticks])
    plt.yticks(range(len(labels)), labels.keys())

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
