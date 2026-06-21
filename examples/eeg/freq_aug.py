"""Augmentations dans le domaine frequentiel pour EEG.

L'idee : les bandes de frequence EEG sont cliniquement significatives.
  delta  0.5-4 Hz  — sommeil profond, coma
  theta  4-8  Hz   — somnolence, epilepsie temporale
  alpha  8-13 Hz   — relaxation, yeux fermes
  beta  13-30 Hz   — eveil, anxiete
  gamma 30+   Hz   — traitement cognitif, crises focales

Masquer une bande entiere force l'encodeur a apprendre des representations
specifiques a chaque bande — impossible avec du masquage temporel brut.
C'est l'equivalent de masquer des "tokens semantiques" pour l'EEG.

Usage dans le SSL :
    x_aug = freq_band_mask(x, sfreq=200, n_bands_to_mask=1)
    # x_aug : meme forme que x, bande(s) aleatoire(s) annulees dans Fourier
"""
import torch
import torch.nn.functional as F


# Bandes standard (Hz) — ordre clinique
BANDS = {
    "delta": (0.5,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 80.0),
}
BAND_NAMES = list(BANDS.keys())


def _hz_to_bin(hz: float, sfreq: float, n_fft: int) -> int:
    """Convertit une frequence Hz en indice FFT."""
    return int(round(hz * n_fft / sfreq))


def freq_band_mask(
    x: torch.Tensor,
    sfreq: float = 200.0,
    n_bands_to_mask: int = 1,
    band_names: list | None = None,
) -> torch.Tensor:
    """Annule aleatoirement n bandes de frequence dans x: [B, C, T].

    Procedure :
      1. rFFT sur la dimension temporelle  -> [B, C, T//2+1] complexe
      2. Annuler les bins correspondant aux bandes choisies
      3. irFFT -> [B, C, T]  (meme longueur que l'entree)

    Args:
        x              : signal EEG [B, C, T], float32
        sfreq          : frequence d'echantillonnage en Hz
        n_bands_to_mask: nombre de bandes a masquer (1 ou 2)
        band_names     : si None, choisit aleatoirement parmi BAND_NAMES

    Returns:
        x_masked : meme shape que x
    """
    B, C, T = x.shape

    # Choisir les bandes a masquer
    if band_names is None:
        idx = torch.randperm(len(BAND_NAMES))[:n_bands_to_mask].tolist()
        chosen = [BAND_NAMES[i] for i in idx]
    else:
        chosen = band_names

    # FFT
    X_f = torch.fft.rfft(x, n=T, dim=-1)       # [B, C, T//2+1] complex

    n_fft = T // 2 + 1
    for name in chosen:
        lo, hi = BANDS[name]
        bin_lo = _hz_to_bin(lo, sfreq, T)
        bin_hi = _hz_to_bin(hi, sfreq, T) + 1
        bin_lo = max(0, bin_lo)
        bin_hi = min(n_fft, bin_hi)
        if bin_lo < bin_hi:
            X_f[:, :, bin_lo:bin_hi] = 0.0

    # iFFT -> signal reel de meme longueur
    return torch.fft.irfft(X_f, n=T, dim=-1)


def random_freq_aug(
    x: torch.Tensor,
    sfreq: float = 200.0,
    p_mask: float = 0.5,
    n_bands: int = 1,
) -> torch.Tensor:
    """Applique freq_band_mask avec probabilite p_mask, sinon retourne x."""
    if torch.rand(1).item() < p_mask:
        return freq_band_mask(x, sfreq=sfreq, n_bands_to_mask=n_bands)
    return x
