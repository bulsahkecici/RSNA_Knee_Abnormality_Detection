"""Typed pipeline errors with Turkish action hints."""

from __future__ import annotations


class RsnaError(Exception):
    code: str = "RSNA"
    action_tr: str = "Durumu rsna status ile kontrol edin."

    def __init__(self, message: str, *, action_tr: str | None = None, code: str | None = None):
        super().__init__(message)
        if action_tr:
            self.action_tr = action_tr
        if code:
            self.code = code

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": str(self), "action_tr": self.action_tr}


class NeedsAuthError(RsnaError):
    code = "NEEDS_AUTH"
    action_tr = "Kaggle veya Colab oturumunu resmi araçla açın; credential kopyalamayın."


class NeedsRuntimeError(RsnaError):
    code = "NEEDS_RUNTIME"
    action_tr = "rsna runtime capabilities çıktısındaki handoff adımlarını uygulayın."


class QuarantineError(RsnaError):
    code = "QUARANTINE"
    action_tr = "Quarantine kaydını okuyun; sahte etiket üretmeyin."


class LeakageError(RsnaError):
    code = "LEAKAGE"
    action_tr = "folds.csv ve gold eval ayrımını rsna audit ile doğrulayın."


class ContractError(RsnaError):
    code = "CONTRACT"
    action_tr = "Şema veya submission sözleşmesini onarın; sahte skor yazmayın."


class FakeResultError(RsnaError):
    code = "FAKE_RESULT"
    action_tr = "Simülatör veya sentetik AUC production gate'e giremez. Kaynağı düzeltin."
