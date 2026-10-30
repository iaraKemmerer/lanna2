import numpy as np
from scipy.signal import butter, sosfiltfilt

from config_lanna2 import CFG


class ChannelConditions:
    """
    The audio conditions ("forms") compared in the paper. They act on the whole 16-kHz file; the vowel is cut
    afterwards (AudioDataProcessor.network_input).

        original        the recording as it is
        dc              DC offset removed
        pol             canonical polarity: the file is flipped if the vowel is skewed downwards
        dcpol           DC removed + canonical polarity
        noise_for_all   DC removed, noise added to EVERY recording 5 dB below its vowel, canonical polarity
        donor_v3        (discarded) SLI recordings get the noise and SNR of a random training TD "donor";
                        it needs the label, also at test time
        adaptive_v4     (discarded) 80 Hz-7 kHz band-pass, noise added only up to a common SNR, polarity, RMS

    None of them uses the label except donor_v3. Every noise is synthesized from silences of TRAINING children of
    the fold (never validation or test), which is why an instance belongs to one fold. Each recording gets its own
    random generator, np.random.default_rng(1000 * patient_id + ord(vowel)), so it always receives the same noise.
    """

    def __init__(self, processor, train_ids):
        """processor: AudioDataProcessor; train_ids: patient ids of the training children of the fold."""
        self.processor = processor
        files = processor.list_files()
        self.td_train = [f for f in files if processor.group_of(f) == "TD" and processor.patient_id(f) in train_ids]
        self.sli_train = [f for f in files if processor.group_of(f) == "SLI" and processor.patient_id(f) in train_ids]
        self._bank, self._donors, self._v4 = None, None, None

    # =============================================================
    # SIGNAL TOOLS
    # =============================================================
    @staticmethod
    def power(x):
        return float(np.mean(x ** 2)) if len(x) else 0.0

    @staticmethod
    def synth_noise(bank, n, rng):
        """
        n samples of stationary noise with unit RMS: overlap-add (50%) of 32-ms Hann-windowed fragments taken at
        random from the silence segments in `bank`. It keeps the spectral color of the silences.
        """
        frag = CFG.NOISE_FRAGMENT
        hop, win = frag // 2, np.hanning(frag)
        out = np.zeros(n + frag)
        for s in range(0, n + 1, hop):
            seg = bank[rng.integers(len(bank))]
            a = rng.integers(0, len(seg) - frag + 1)
            out[s:s + frag] += seg[a:a + frag] * win
        out = out[:n] - out[:n].mean()
        return out / (np.sqrt(np.mean(out ** 2)) + 1e-12)

    @staticmethod
    def canonical_polarity(w, start, end):
        """
        Flips the whole file if the vowel is skewed downwards (skewness < 0). In LANNA /o/, 69% of SLI and 33% of TD
        vowels have positive skewness, so polarity separates the classes; this makes it the same for everyone.
        """
        v = w[start:end] - w[start:end].mean()
        skew = np.mean(v ** 3) / (np.mean(v ** 2) ** 1.5 + 1e-12)
        return -w if skew < 0 else w

    def own_snr(self, w, start, end):
        """(noise power, SNR in dB) of a recording, measured on its own silence; SNR is inf without silence."""
        sil = self.processor.silence_parts(w, start, end)
        pn = self.power(np.concatenate(sil)) if sil else 0.0
        return pn, (10 * np.log10(self.power(w[start:end]) / pn) if pn > 0 else np.inf)

    @staticmethod
    def bandpass(w):
        """4th-order Butterworth band-pass (CFG.V4_BAND) applied forwards and backwards (zero phase)."""
        sos = butter(4, CFG.V4_BAND, btype="band", fs=CFG.SAMPLE_RATE, output="sos")
        return np.ascontiguousarray(sosfiltfilt(sos, w - w.mean()))

    # =============================================================
    # NOISE BANKS OF THE FOLD (training children only)
    # =============================================================
    def bank(self):
        """Raw silences of the training TD children, all five vowels: the noise of noise_for_all."""
        if self._bank is None:
            self._bank = []
            for f in self.td_train:
                self._bank += self.processor.silence_parts(*self.processor.load_recording(f))
        return self._bank

    def donors(self):
        """{vowel: [(silences, snr_db), ...]}: training TD recordings with silence (donor_v3)."""
        if self._donors is None:
            self._donors = {}
            for f in self.td_train:
                w, s, e = self.processor.load_recording(f)
                w = w - w.mean()
                sil = self.processor.silence_parts(w, s, e)
                if sil:
                    snr = 10 * np.log10(self.power(w[s:e]) / self.power(np.concatenate(sil)))
                    self._donors.setdefault(self.processor.vowel_of(f), []).append((sil, snr))
        return self._donors

    def v4(self):
        """(band-passed silences of training TD + SLI children, common SNR) for adaptive_v4."""
        if self._v4 is None:
            bank, snrs = [], []
            for f in self.td_train + self.sli_train:
                w, s, e = self.processor.load_recording(f)
                w1 = self.bandpass(w)
                bank += self.processor.silence_parts(w1, s, e)
                snr = self.own_snr(w1, s, e)[1]
                if np.isfinite(snr):
                    snrs.append(snr)
            self._v4 = (bank, float(np.percentile(snrs, CFG.V4_SNR_PERCENTILE)))
        return self._v4

    # =============================================================
    # MAIN FUNCTION
    # =============================================================
    def apply(self, condition, w, start, end, patient_id, vowel, group, snr_db=None):
        """Returns (processed whole file, normalization of the vowel: 'peak' or 'rms')."""
        snr_db = CFG.NOISE_SNR_DB if snr_db is None else snr_db
        rng = np.random.default_rng(1000 * patient_id + ord(vowel))
        if condition == "original":
            return w, "peak"
        if condition == "dc":
            return w - w.mean(), "peak"
        if condition == "pol":
            return self.canonical_polarity(w, start, end), "peak"
        if condition in ("dcpol", "noise_for_all"):
            x = w - w.mean()
            if condition == "noise_for_all":
                a = np.sqrt(self.power(x[start:end]))
                x = x + a * 10 ** (-snr_db / 20) * self.synth_noise(self.bank(), len(x), rng)
            x = x - x.mean()
            return self.canonical_polarity(x, start, end), "peak"
        if condition == "donor_v3":
            x = w - w.mean()
            if group == "SLI":
                donors = self.donors()[vowel]
                sil_d, snr_d = donors[rng.integers(len(donors))]
                pn0 = self.own_snr(x, start, end)[0]
                pa = max(self.power(x[start:end]) / 10 ** (snr_d / 10) - pn0, 0.0)
                x = x + np.sqrt(pa) * self.synth_noise(sil_d, len(x), rng)
                x = x - x.mean()
            return self.canonical_polarity(x, start, end), "peak"
        if condition == "adaptive_v4":
            bank_bp, snr_common = self.v4()
            x = self.bandpass(w)
            pn0 = self.own_snr(x, start, end)[0]
            pa = max(self.power(x[start:end]) / 10 ** (snr_common / 10) - pn0, 0.0)
            if pa > 0:
                x = x + np.sqrt(pa) * self.synth_noise(bank_bp, len(x), rng)
            return self.canonical_polarity(x, start, end), "rms"
        raise ValueError(f"unknown condition {condition!r}; choose from {CFG.CONDITIONS}")
