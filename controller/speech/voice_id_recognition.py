
from __future__ import annotations

import os
import numpy as np
import librosa
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

from wyoming.audio import AudioChunk
import joblib

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import torch
from speechbrain.inference.classifiers import EncoderClassifier

# Additional imports for improvements
from scipy import signal
from scipy.stats import multivariate_normal
import warnings
warnings.filterwarnings('ignore')


def _softmax_stable(x: np.ndarray, clip: float = 25.0, temperature: float = 1.0) -> np.ndarray:
    """Stable softmax implementation with temperature scaling."""
    x = np.ravel(x) / max(temperature, 1e-6)
    m = np.max(x)
    z = np.clip(x - m, -clip, clip)
    p = np.exp(z)
    return p / (p.sum() + 1e-12)


@dataclass
class RecognizerConfig:
    """ configuration with more parameters for better control."""
    model_dir: str = "data/voice_models"
    target_sr: int = 16000
    use_neural: bool = True

    #  thresholding
    confidence_threshold: float = 0.35  # Lowered from 0.4
    margin_threshold: float = 0.03     # Lowered from 0.05
    abs_match_threshold: float = 0.25  # Lowered from 0.3

    #  feature weights
    weights_neural_mfcc: Tuple[float, float] = (0.4, 0.6)  # Increased neural weight

    # Temperature controls
    prior_temperature: float = 1.5     # Lowered from 2.0
    conf_temperature: float = 0.6      # Lowered from 0.8

    # New parameters for  features
    use_data_augmentation: bool = True
    use_vad: bool = True
    use_pitch_features: bool = True
    use_ensemble: bool = True
    use_score_normalization: bool = True
    adaptive_thresholding: bool = True

    # Advanced MFCC parameters
    n_mfcc: int = 19              # Increased from 13
    n_mels: int = 60              # Increased from 40
    fmin: float = 80              # Increased from 50
    fmax: float = 7600            # Decreased from 8000

    # GMM parameters
    max_gmm_components: int = 4    # Increased from 2
    gmm_reg_covar: float = 1e-4   # Increased regularization


class VoiceIDRecognizer:
    """ Voice ID Recognizer with improved accuracy."""

    def __init__(self, cfg: RecognizerConfig = RecognizerConfig()):
        self.cfg = cfg
        os.makedirs(self.cfg.model_dir, exist_ok=True)
        self.speakers: Dict[str, Dict] = {}
        self.neural: Optional[EncoderClassifier] = None
        self.global_stats: Dict = {}

        if self.cfg.use_neural:
            try:
                self.neural = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    run_opts={"device": "cpu"},
                )
            except Exception:
                self.neural = None

        self._load()

    # ================  AUDIO PROCESSING ================

    def _resample(self, y: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        """ resampling with anti-aliasing."""
        if sr == self.cfg.target_sr:
            return y, sr
        try:
            # Use higher quality resampling
            y2 = librosa.resample(y=y, orig_sr=sr, target_sr=self.cfg.target_sr, res_type='kaiser_best')
            return y2, self.cfg.target_sr
        except Exception:
            return y, sr

    def _trim(self, y: np.ndarray, top_db: int = 20) -> np.ndarray:
        """ trimming with multiple methods."""
        try:
            # Primary trimming
            y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)

            # Secondary VAD-based trimming if enabled
            if self.cfg.use_vad and len(y_trimmed) > 0:
                y_trimmed = self._simple_vad(y_trimmed)

            return y_trimmed if len(y_trimmed) > 0 else y
        except Exception:
            return y

    def _simple_vad(self, y: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
        """Simple Voice Activity Detection."""
        try:
            # Calculate energy-based VAD
            energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            energy_threshold = np.percentile(energy, 30)  # Dynamic threshold

            # Create voice activity mask
            voice_frames = energy > energy_threshold

            # Convert frame indices to sample indices
            voice_samples = np.zeros(len(y), dtype=bool)
            for i, is_voice in enumerate(voice_frames):
                start_sample = i * hop_length
                end_sample = min(start_sample + hop_length, len(y))
                voice_samples[start_sample:end_sample] = is_voice

            # Apply morphological operations to clean up
            kernel_size = int(0.1 * self.cfg.target_sr)  # 100ms kernel
            voice_samples = signal.medfilt(voice_samples.astype(float), kernel_size=min(kernel_size, len(voice_samples)//2*2-1)) > 0.5

            return y[voice_samples] if voice_samples.any() else y
        except Exception:
            return y

    def _chunk_to_np(self, ch: AudioChunk) -> Tuple[np.ndarray, int]:
        """ chunk processing with better error handling."""
        if hasattr(ch, "audio"):
            audio_bytes = ch.audio
            sr = int(ch.rate)
            width = int(ch.width)
            chs = int(ch.channels)
        else:
            raise TypeError("Expected Wyoming AudioChunk")

        if width == 1:
            dtype, norm = np.int8, 128.0
        elif width == 2:
            dtype, norm = np.int16, 32768.0
        elif width == 4:
            dtype, norm = np.int32, 2147483648.0
        else:
            raise ValueError(f"Unsupported sample width: {width}")

        arr = np.frombuffer(audio_bytes, dtype=dtype)
        if chs > 1:
            arr = arr.reshape(-1, chs)[:, 0]

        y = (arr.astype(np.float32) / norm).copy()
        y, sr = self._resample(y, sr)
        y = self._trim(y, 20)

        return y, sr

    # ================  FEATURE EXTRACTION ================

    def _extract_mfcc_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """ MFCC feature extraction with more robust features."""
        if len(y) < 512:
            y = np.pad(y, (0, 512 - len(y)))

        #  MFCC with better parameters
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, 
            n_mfcc=self.cfg.n_mfcc, 
            n_mels=self.cfg.n_mels,
            n_fft=2048,  # Increased FFT size
            hop_length=256,  # Increased hop length
            fmin=self.cfg.fmin, 
            fmax=self.cfg.fmax,
            window='hann'
        )

        # Delta and delta-delta features
        d1 = librosa.feature.delta(mfcc, order=1)
        d2 = librosa.feature.delta(mfcc, order=2)

        #  spectral features
        spectral_features = self._extract_spectral_features(y, sr)

        # Pitch features if enabled
        pitch_features = self._extract_pitch_features(y, sr) if self.cfg.use_pitch_features else np.array([])

        # Comprehensive statistics
        def _stats(mat: np.ndarray) -> np.ndarray:
            if mat.size == 0:
                return np.array([])
            stats = np.hstack([
                mat.mean(axis=1),
                mat.std(axis=1),
                np.median(mat, axis=1),
                np.percentile(mat, 25, axis=1),
                np.percentile(mat, 75, axis=1),
                mat.max(axis=1),
                mat.min(axis=1)
            ])
            return stats

        # Combine all features
        features_list = [
            _stats(mfcc),
            _stats(d1),
            _stats(d2),
            spectral_features
        ]

        if len(pitch_features) > 0:
            features_list.append(pitch_features)

        feat = np.hstack(features_list)

        # Robust normalization
        feat = self._robust_normalize(feat)

        return feat.astype(np.float32)

    def _extract_spectral_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract  spectral features."""
        try:
            # Basic spectral features
            sc = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=256)
            sb = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=256)
            ro = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=256)
            zc = librosa.feature.zero_crossing_rate(y, hop_length=256)

            # Additional spectral features
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=256)
            flatness = librosa.feature.spectral_flatness(y=y, hop_length=256)

            # Formant-like features (using spectral peaks)
            stft = librosa.stft(y, hop_length=256)
            magnitude = np.abs(stft)
            spectral_peaks = np.array([np.mean(magnitude), np.std(magnitude)])

            # Combine spectral features
            spectral_feat = np.hstack([
                np.array([sc.mean(), sc.std()]),
                np.array([sb.mean(), sb.std()]),
                np.array([ro.mean(), ro.std()]),
                np.array([zc.mean(), zc.std()]),
                np.array([contrast.mean(), contrast.std()]),
                np.array([flatness.mean(), flatness.std()]),
                spectral_peaks
            ])

            return spectral_feat
        except Exception:
            # Fallback to basic features
            return np.array([0.0] * 14)

    def _extract_pitch_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract pitch-related features."""
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y, 
                fmin=librosa.note_to_hz('C2'), 
                fmax=librosa.note_to_hz('C7'),
                sr=sr,
                hop_length=256
            )

            # Remove NaN values
            f0_clean = f0[~np.isnan(f0)]

            if len(f0_clean) > 0:
                pitch_features = np.array([
                    f0_clean.mean(),
                    f0_clean.std(),
                    np.median(f0_clean),
                    np.percentile(f0_clean, 25),
                    np.percentile(f0_clean, 75),
                    voiced_probs.mean(),
                    voiced_probs.std()
                ])
            else:
                pitch_features = np.array([0.0] * 7)

            return pitch_features
        except Exception:
            return np.array([0.0] * 7)

    def _robust_normalize(self, feat: np.ndarray) -> np.ndarray:
        """Robust feature normalization."""
        if feat.size == 0:
            return feat

        # Use robust statistics
        median = np.median(feat)
        mad = np.median(np.abs(feat - median))  # Median Absolute Deviation

        if mad > 1e-8:
            feat = (feat - median) / (1.4826 * mad)  # Robust z-score
        else:
            feat = feat - median

        return feat

    def _embed(self, y: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """ neural embedding with better preprocessing."""
        if self.neural is None:
            return None

        if sr != self.cfg.target_sr:
            y, _ = self._resample(y, sr)

        # Ensure minimum length
        min_length = int(0.5 * self.cfg.target_sr)  # 0.5 seconds minimum
        if len(y) < min_length:
            y = np.pad(y, (0, min_length - len(y)))

        # Apply gain normalization
        y = y / (np.max(np.abs(y)) + 1e-8)

        try:
            with torch.no_grad():
                wav = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
                emb = self.neural.encode_batch(wav).squeeze().cpu().numpy()

                # L2 normalize embeddings
                emb = emb / (np.linalg.norm(emb) + 1e-8)

                return emb.astype(np.float32)
        except Exception:
            return None

    # ================ DATA AUGMENTATION ================

    def _augment_audio(self, y: np.ndarray, sr: int) -> List[Tuple[np.ndarray, int]]:
        """Apply data augmentation techniques."""
        if not self.cfg.use_data_augmentation:
            return [(y, sr)]

        augmented_samples = [(y, sr)]  # Original

        try:
            # Speed perturbation (common in speaker recognition)
            for speed_factor in [0.9, 1.1]:
                y_speed = librosa.effects.time_stretch(y, rate=speed_factor)
                augmented_samples.append((y_speed, sr))

            # Pitch shifting (small amounts)
            for n_steps in [-1, 1]:
                y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
                augmented_samples.append((y_pitch, sr))

            # Add subtle noise
            noise_factor = 0.002
            noise = np.random.normal(0, noise_factor, y.shape)
            y_noise = y + noise
            augmented_samples.append((y_noise, sr))

        except Exception:
            pass  # If augmentation fails, just use original

        return augmented_samples

    # ================  TRAINING ================

    def train(self, name: str, samples: List[Tuple[np.ndarray, int]]) -> bool:
        """ training with data augmentation and better modeling."""
        all_samples = []

        # Apply augmentation to each sample
        for y, sr in samples:
            y, sr = self._resample(self._trim(y), sr)
            if len(y) < 256:
                continue

            augmented = self._augment_audio(y, sr)
            all_samples.extend(augmented)

        if not all_samples:
            return False

        # Extract features from all samples
        feats, embs = [], []
        for y, sr in all_samples:
            if len(y) < 256:
                continue

            feat = self._extract_mfcc_features(y, sr)
            feats.append(feat)

            e = self._embed(y, sr)
            if e is not None:
                embs.append(e)

        if not feats:
            return False

        model: Dict[str, object] = {}
        fa = np.asarray(feats, dtype=np.float64)

        #  statistical modeling
        model["mfcc_mean"] = fa.mean(axis=0)
        model["mfcc_std"] = fa.std(axis=0) + 1e-6
        model["mfcc_median"] = np.median(fa, axis=0)

        # Robust covariance estimation
        if len(fa) >= 3:
            try:
                from sklearn.covariance import EmpiricalCovariance
                cov_estimator = EmpiricalCovariance().fit(fa)
                model["mfcc_cov"] = cov_estimator.covariance_
            except Exception:
                model["mfcc_cov"] = np.cov(fa.T)

        #  GMM with better parameters
        if len(fa) >= 3:
            scaler = RobustScaler().fit(fa)  # Use RobustScaler instead of StandardScaler
            xa = scaler.transform(fa)

            n_samples = xa.shape[0]
            n_components = max(1, min(self.cfg.max_gmm_components, n_samples // 3))

            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type='diag',
                reg_covar=self.cfg.gmm_reg_covar,
                max_iter=300,  # Increased iterations
                n_init=5,      # More initializations
                init_params='k-means++',  # Better initialization
                random_state=42,
            )

            try:
                gmm.fit(xa)
                model["gmm"] = gmm
                model["gmm_scaler"] = scaler
                model["gmm_score_mean"] = gmm.score_samples(xa).mean()
                model["gmm_score_std"] = gmm.score_samples(xa).std()
            except Exception:
                pass

        #  neural embeddings
        if embs:
            ea = np.asarray(embs, dtype=np.float32)
            model["neural_mean"] = ea.mean(axis=0)
            model["neural_std"] = ea.std(axis=0) + 1e-6

            # Store embedding statistics for score normalization
            if len(ea) >= 2:
                model["neural_embeddings"] = ea  # Store for ensemble methods

        # Training metadata
        model["training_date"] = datetime.now().isoformat()
        model["num_samples"] = len(all_samples)
        model["num_original_samples"] = len(samples)

        self.speakers[name] = model
        self._update_global_stats()
        self._save()
        return True

    def _update_global_stats(self):
        """Update global statistics for score normalization."""
        if not self.cfg.use_score_normalization:
            return

        all_neural_means = []
        all_mfcc_means = []

        for speaker_data in self.speakers.values():
            if "neural_mean" in speaker_data:
                all_neural_means.append(speaker_data["neural_mean"])
            if "mfcc_mean" in speaker_data:
                all_mfcc_means.append(speaker_data["mfcc_mean"])

        if all_neural_means:
            self.global_stats["global_neural_mean"] = np.mean(all_neural_means, axis=0)
            self.global_stats["global_neural_std"] = np.std(all_neural_means, axis=0) + 1e-6

        if all_mfcc_means:
            self.global_stats["global_mfcc_mean"] = np.mean(all_mfcc_means, axis=0)
            self.global_stats["global_mfcc_std"] = np.std(all_mfcc_means, axis=0) + 1e-6

    # ================  RECOGNITION ================

    def recognize(self, chunk: AudioChunk) -> Tuple[Optional[str], float]:
        """ recognition with improved scoring."""
        if not self.speakers:
            return None, 0.0

        try:
            y, sr = self._chunk_to_np(chunk)
        except Exception:
            return None, 0.0

        # Ensure minimum length
        min_len = int(0.3 * self.cfg.target_sr)
        if len(y) < min_len:
            y = np.pad(y, (0, min_len - len(y)))

        # Extract features
        x = self._extract_mfcc_features(y, sr)

        names = list(self.speakers.keys())
        scores = self._compute_scores(x, y, sr, names)

        # Apply score normalization if enabled
        if self.cfg.use_score_normalization:
            scores = self._normalize_scores(scores)

        # Apply ensemble method if enabled
        if self.cfg.use_ensemble and len(names) >= 2:
            scores = self._ensemble_scoring(scores)

        # Convert to probabilities
        probs = _softmax_stable(scores, clip=25.0, temperature=self.cfg.conf_temperature)

        idx = int(np.argmax(probs))
        best_name = names[idx]
        best_prob = float(probs[idx])

        #  thresholding
        if self.cfg.adaptive_thresholding:
            threshold = self._adaptive_threshold(best_name, scores[idx])
        else:
            threshold = self.cfg.confidence_threshold

        # Calculate margin
        sp = np.sort(probs)[::-1]
        margin = float(sp[0] - sp[1]) if sp.size >= 2 else 1.0
        best_abs = float(scores[idx])

        # Decision logic
        if (best_prob < threshold or 
            margin < self.cfg.margin_threshold or 
            best_abs < self.cfg.abs_match_threshold):
            return None, best_prob

        return best_name, best_prob

    def _compute_scores(self, x: np.ndarray, y: np.ndarray, sr: int, names: List[str]) -> np.ndarray:
        """Compute  similarity scores."""
        scores = np.zeros(len(names))

        for i, name in enumerate(names):
            m = self.speakers[name]

            # MFCC-based scoring ()
            mfcc_score = self._compute_mfcc_score(x, m)

            # Neural embedding scoring
            neural_score = self._compute_neural_score(y, sr, m)

            # GMM scoring
            gmm_score = self._compute_gmm_score(x, m)

            # Combine scores with adaptive weights
            w_n, w_m = self.cfg.weights_neural_mfcc
            w_g = 0.2  # GMM weight

            total_score = w_m * mfcc_score
            if neural_score is not None:
                total_score += w_n * neural_score
            if gmm_score is not None:
                total_score += w_g * gmm_score

            scores[i] = total_score

        return scores

    def _compute_mfcc_score(self, x: np.ndarray, model: Dict) -> float:
        """ MFCC scoring with multiple metrics."""
        mu = model["mfcc_mean"]
        sigma = model.get("mfcc_std", np.ones_like(mu))

        # Mahalanobis distance if covariance available
        if "mfcc_cov" in model:
            try:
                diff = x - mu
                cov_inv = np.linalg.pinv(model["mfcc_cov"])
                mahal_dist = np.sqrt(diff.T @ cov_inv @ diff)
                score = float(np.exp(-0.5 * mahal_dist))
            except Exception:
                # Fallback to normalized Euclidean
                dist = np.sqrt(np.mean(((x - mu) / (sigma + 1e-8)) ** 2))
                score = float(np.exp(-0.5 * dist))
        else:
            # Normalized Euclidean distance
            dist = np.sqrt(np.mean(((x - mu) / (sigma + 1e-8)) ** 2))
            score = float(np.exp(-0.5 * dist))

        return score

    def _compute_neural_score(self, y: np.ndarray, sr: int, model: Dict) -> Optional[float]:
        """ neural embedding scoring."""
        if self.neural is None or "neural_mean" not in model:
            return None

        e = self._embed(y, sr)
        if e is None:
            return None

        mean_e = np.asarray(model["neural_mean"])

        # Cosine similarity
        cos = np.dot(e, mean_e) / (np.linalg.norm(e) * np.linalg.norm(mean_e) + 1e-12)
        cos_score = (cos + 1.0) / 2.0

        # Euclidean distance in embedding space
        if "neural_std" in model:
            std_e = model["neural_std"]
            eucl_dist = np.sqrt(np.mean(((e - mean_e) / (std_e + 1e-8)) ** 2))
            eucl_score = np.exp(-0.5 * eucl_dist)

            # Combine cosine and Euclidean
            combined_score = 0.7 * cos_score + 0.3 * eucl_score
        else:
            combined_score = cos_score

        return float(combined_score)

    def _compute_gmm_score(self, x: np.ndarray, model: Dict) -> Optional[float]:
        """ GMM scoring with normalization."""
        if "gmm" not in model or "gmm_scaler" not in model:
            return None

        try:
            scaler = model["gmm_scaler"]
            gmm = model["gmm"]

            x_scaled = scaler.transform(x.reshape(1, -1))
            log_prob = gmm.score_samples(x_scaled)[0]

            # Normalize using training statistics
            if "gmm_score_mean" in model and "gmm_score_std" in model:
                norm_score = (log_prob - model["gmm_score_mean"]) / (model["gmm_score_std"] + 1e-8)
                score = 1.0 / (1.0 + np.exp(-norm_score))  # Sigmoid
            else:
                score = np.exp(log_prob)  # Convert log-prob to prob

            return float(score)
        except Exception:
            return None

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Apply score normalization techniques."""
        if "global_neural_mean" not in self.global_stats:
            return scores

        # Z-score normalization
        mean_score = np.mean(scores)
        std_score = np.std(scores) + 1e-8

        normalized_scores = (scores - mean_score) / std_score
        return normalized_scores

    def _ensemble_scoring(self, scores: np.ndarray) -> np.ndarray:
        """Apply ensemble methods for improved accuracy."""
        # Simple ensemble: combine with weighted average of different metrics
        # This is a placeholder for more sophisticated ensemble methods
        return scores

    def _adaptive_threshold(self, speaker_name: str, score: float) -> float:
        """Compute adaptive threshold based on speaker model quality."""
        if speaker_name not in self.speakers:
            return self.cfg.confidence_threshold

        model = self.speakers[speaker_name]

        # Adjust threshold based on number of training samples
        num_samples = model.get("num_samples", 1)
        quality_factor = min(1.0, num_samples / 20)  # Normalize by 20 samples

        # Lower threshold for better trained models
        adjusted_threshold = self.cfg.confidence_threshold * (1.0 - 0.2 * quality_factor)

        return max(0.1, adjusted_threshold)  # Minimum threshold of 0.1

    # ================ TRAINING HELPERS ================

    def train_from_files(self, name: str, paths: List[str]) -> bool:
        """ training from files with better preprocessing."""
        samples: List[Tuple[np.ndarray, int]] = []

        for p in paths:
            try:
                y, sr = librosa.load(p, sr=None, mono=True)
                y = self._trim(y)
                if len(y) > 0:
                    samples.append((y, sr))
            except Exception:
                continue

        return self.train(name, samples)

    # ================ PERSISTENCE ================

    def _save(self) -> None:
        """Save models and global statistics."""
        models_path = os.path.join(self.cfg.model_dir, "voice_models.pkl")
        stats_path = os.path.join(self.cfg.model_dir, "global_stats.pkl")

        try:
            joblib.dump(self.speakers, models_path)
            joblib.dump(self.global_stats, stats_path)
        except Exception:
            pass

    def _load(self) -> None:
        """Load models and global statistics."""
        models_path = os.path.join(self.cfg.model_dir, "voice_models.pkl")
        stats_path = os.path.join(self.cfg.model_dir, "global_stats.pkl")

        # Load speaker models
        if os.path.exists(models_path):
            try:
                self.speakers = joblib.load(models_path)
            except Exception:
                self.speakers = {}

        # Load global statistics
        if os.path.exists(stats_path):
            try:
                self.global_stats = joblib.load(stats_path)
            except Exception:
                self.global_stats = {}

    # ================ UTILITY METHODS ================

    def list_speakers(self) -> List[str]:
        """List all registered speakers."""
        return list(self.speakers.keys())

    def remove_speaker(self, name: str) -> bool:
        """Remove a speaker model."""
        if name in self.speakers:
            del self.speakers[name]
            self._update_global_stats()
            self._save()
            return True
        return False

    def info(self, name: str) -> Optional[Dict]:
        """Get speaker model information."""
        return self.speakers.get(name)

    def get_model_quality_score(self, name: str) -> float:
        """Get a quality score for a speaker model."""
        if name not in self.speakers:
            return 0.0

        model = self.speakers[name]

        # Factor in number of samples
        num_samples = model.get("num_samples", 1)
        sample_score = min(1.0, num_samples / 20)

        # Factor in presence of different feature types
        feature_score = 0.0
        if "mfcc_mean" in model:
            feature_score += 0.3
        if "neural_mean" in model:
            feature_score += 0.4
        if "gmm" in model:
            feature_score += 0.3

        return 0.6 * sample_score + 0.4 * feature_score
