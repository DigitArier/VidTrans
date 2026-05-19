from __future__ import annotations

import logging
import types

import torch
import transformers.pytorch_utils as _pt_utils

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patch 1: isin_mps_friendly
# ---------------------------------------------------------------------------
# Ursprung: transformers.pytorch_utils (entfernt in transformers>=4.43 / 5.x)
# Verwendung in: TTS/tts/layers/tortoise/autoregressive.py:12
# Original-Semantik: torch.isin mit MPS-Workaround.
# Auf Windows/CUDA (kein MPS vorhanden): direktes torch.isin ist korrekt.
# ---------------------------------------------------------------------------

def _isin_mps_friendly(
    elements: torch.Tensor,
    test_elements: int | torch.Tensor,
) -> torch.Tensor:
    """Ersatz für ``transformers.pytorch_utils.isin_mps_friendly``.

    Auf CUDA/CPU (kein Apple MPS) ist ``torch.isin`` direkt nutzbar.
    ``test_elements`` wird automatisch in einen Tensor auf dem richtigen
    Device konvertiert, falls ein skalarer ``int`` übergeben wird.

    Args:
        elements: Eingabe-Tensor, dessen Elemente geprüft werden.
        test_elements: Einzelner int oder Tensor mit Vergleichswerten.

    Returns:
        Bool-Tensor gleicher Form wie ``elements``.

    Raises:
        TypeError: Wenn ``test_elements`` kein int oder Tensor ist.
    """
    if isinstance(test_elements, int):
        # torch.isin erwartet einen Tensor als test_elements
        test_elements = torch.tensor(
            [test_elements],
            dtype=elements.dtype,
            device=elements.device,
        )
    elif isinstance(test_elements, torch.Tensor):
        # Sicherstellen, dass test_elements auf demselben Device liegt
        test_elements = test_elements.to(elements.device)
    else:
        raise TypeError(
            f"test_elements muss int oder torch.Tensor sein, "
            f"nicht {type(test_elements)}"
        )
    return torch.isin(elements, test_elements)

def _apply_isin_mps_friendly_patch() -> None:
    """Injiziert ``isin_mps_friendly`` in ``transformers.pytorch_utils``."""
    if hasattr(_pt_utils, "isin_mps_friendly"):
        # Bereits vorhanden (ältere transformers-Version) – kein Patch nötig
        logger.debug(
            "compat_patches: isin_mps_friendly bereits in "
            "transformers.pytorch_utils vorhanden – kein Patch nötig."
        )
        return

    _pt_utils.isin_mps_friendly = _isin_mps_friendly  # type: ignore[attr-defined]
    logger.info(
        "compat_patches: isin_mps_friendly erfolgreich in "
        "transformers.pytorch_utils injiziert (transformers>=4.43 Patch)."
    )


# ---------------------------------------------------------------------------
# Patch 2: deepmultilingualpunctuation – grouped_entities → aggregation_strategy
# ---------------------------------------------------------------------------
# Ursprung: deepmultilingualpunctuation/punctuationmodel.py:9
#   self.pipe = pipeline("ner", model, grouped_entities=False, device=0)
# Problem:  grouped_entities wurde in transformers>=5.0 vollständig entfernt.
# Fix:      PunctuationModel.__init__ wird durch eine Wrapper-Version ersetzt,
#           die aggregation_strategy="none" übergibt (semantisch identisch).
# ---------------------------------------------------------------------------

def _apply_punctuation_model_patch() -> None:
    """Patcht TokenClassificationPipeline für grouped_entities-Kompatibilität.

    Statt PunctuationModel zu ersetzen (scheitert bei direktem from-Import),
    wird _sanitize_parameters der Pipeline selbst gepatcht. grouped_entities
    wird transparent in aggregation_strategy übersetzt – für alle Aufrufer.
    """
    try:
        from transformers.pipelines.token_classification import (
            TokenClassificationPipeline as _TCP,
        )
    except ImportError:
        logger.debug(
            "compat_patches: TokenClassificationPipeline nicht verfügbar "
            "– Patch 2 übersprungen."
        )
        return

    # Idempotenz-Check
    if getattr(_TCP, "_grouped_entities_patched", False):
        logger.debug(
            "compat_patches: TokenClassificationPipeline bereits gepatcht "
            "– übersprungen."
        )
        return

    # Referenz auf die originale Methode sichern
    _original_sanitize = _TCP._sanitize_parameters

    def _patched_sanitize(self, **kwargs):  # type: ignore[override]
        """Wrapper: übersetzt grouped_entities → aggregation_strategy."""
        if "grouped_entities" in kwargs:
            grouped = kwargs.pop("grouped_entities")
            # Nur setzen wenn aggregation_strategy nicht bereits explizit gesetzt
            if "aggregation_strategy" not in kwargs:
                # grouped_entities=False → aggregation_strategy="none"
                # grouped_entities=True  → aggregation_strategy="simple"
                kwargs["aggregation_strategy"] = "none" if not grouped else "simple"
                logger.info(
                    "compat_patches: grouped_entities=%s automatisch zu "
                    "aggregation_strategy='%s' übersetzt.",
                    grouped,
                    kwargs["aggregation_strategy"],
                )
        return _original_sanitize(self, **kwargs)

    # Methode auf der Klasse ersetzen
    _TCP._sanitize_parameters = _patched_sanitize  # type: ignore[method-assign]
    _TCP._grouped_entities_patched = True  # type: ignore[attr-defined]
    logger.info(
        "compat_patches: TokenClassificationPipeline._sanitize_parameters "
        "erfolgreich gepatcht (grouped_entities-Kompatibilität wiederhergestellt)."
    )


# ---------------------------------------------------------------------------
# Patch 3: T5Tokenizer.additional_special_tokens – AttributeError in transformers>=5.x
# ---------------------------------------------------------------------------
# Ursprung: transformers/models/t5/tokenization_t5.py:151 get_sentinel_tokens()
#   self.additional_special_tokens  → AttributeError in transformers>=5.x
# Problem:  additional_special_tokens ist kein direktes Instanz-Attribut mehr.
#           Es liegt in tokenizer.special_tokens_map["additional_special_tokens"]
#           oder tokenizer._added_tokens_encoder (gefilterte Keys).
# Fix:      Property in PreTrainedTokenizerBase injizieren, die den richtigen
#           Accessor-Pfad nimmt – abwärtskompatibel zu transformers<5.
# ---------------------------------------------------------------------------


def _apply_t5_additional_special_tokens_patch() -> None:
    """Patcht PreTrainedTokenizerBase für additional_special_tokens-Kompatibilität.

    Stellt additional_special_tokens als Property bereit, die sowohl unter
    transformers<5 (direktes Attribut) als auch >=5 (special_tokens_map)
    korrekt funktioniert.
    """
    try:
        from transformers.tokenization_utils_base import (
            PreTrainedTokenizerBase as _TokBase,
        )
    except ImportError:
        logger.debug(
            "compat_patches: PreTrainedTokenizerBase nicht verfügbar – "
            "Patch 3 übersprungen."
        )
        return

    # Idempotenz: bereits gepatcht?
    if getattr(_TokBase, "_additional_special_tokens_patched", False):
        logger.debug(
            "compat_patches: additional_special_tokens bereits gepatcht."
        )
        return

    def _get_additional_special_tokens(self) -> list[str]:
        """Gibt additional_special_tokens zurück – kompatibel mit transformers>=5.

        Reihenfolge der Fallback-Pfade:
        1. _additional_special_tokens  (transformers<5, internes Attribut)
        2. special_tokens_map          (transformers>=5, bevorzugter Accessor)
        3. _added_tokens_encoder-Keys  (Fallback: alle hinzugefügten Tokens)
        """
        # Pfad 1: altes internes Attribut noch vorhanden
        internal = object.__getattribute__(self, "__dict__").get(
            "_additional_special_tokens"
        )
        if internal is not None:
            return internal

        # Pfad 2: special_tokens_map (transformers>=5 Standard)
        stm = object.__getattribute__(self, "__dict__").get(
            "_special_tokens_map_extended", {}
        )
        if "additional_special_tokens" in stm:
            return stm["additional_special_tokens"]

        # Pfad 3: special_tokens_map Property wenn vorhanden
        try:
            stm2 = object.__getattribute__(self, "special_tokens_map")
            if "additional_special_tokens" in stm2:
                return stm2["additional_special_tokens"]
        except Exception:
            pass

        # Pfad 4: _added_tokens_encoder gefiltert (letzte Option)
        try:
            added = object.__getattribute__(self, "_added_tokens_encoder")
            # Filtere nach <extra_id_N>-Mustern die das Skript erwartet
            import re as _re
            result = [
                tok for tok in added.keys()
                if _re.search(r"<extra_id_\d+>", tok) is not None
            ]
            if result:
                return result
        except Exception:
            pass

        return []

    def _set_additional_special_tokens(self, value: list[str]) -> None:
        """Setter – speichert additional_special_tokens intern."""
        self.__dict__["_additional_special_tokens"] = value

    # Property auf der Basisklasse setzen – gilt für T5Tokenizer UND T5TokenizerFast
    _TokBase.additional_special_tokens = property(  # type: ignore[attr-defined]
        _get_additional_special_tokens,
        _set_additional_special_tokens,
    )
    _TokBase._additional_special_tokens_patched = True  # type: ignore[attr-defined]
    logger.info(
        "compat_patches: PreTrainedTokenizerBase.additional_special_tokens "
        "als Property gepatcht (transformers>=5.x Patch 3)."
    )


# ---------------------------------------------------------------------------
# Alle Patches beim Import dieses Moduls automatisch anwenden
# ---------------------------------------------------------------------------

def apply_all_patches() -> None:
    """Wendet alle Kompatibilitäts-Patches an.

    Muss VOR dem ersten ``import TTS``, ``PunctuationModel()``
    und ``T5Tokenizer.from_pretrained()`` aufgerufen werden.
    """
    _apply_isin_mps_friendly_patch()              # Patch 1: transformers>=4.43/5.x
    _apply_punctuation_model_patch()              # Patch 2: grouped_entities-Compat
    _apply_t5_additional_special_tokens_patch()   # Patch 3: T5 additional_special_tokens


# Automatische Anwendung beim Import des Moduls
apply_all_patches()