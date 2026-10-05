def __getattr__(name):
    if name == "CalligrapherRecognizer":
        from .recognizer import CalligrapherRecognizer
        return CalligrapherRecognizer
    if name == "EnsembleRecognizer":
        from .ensemble import EnsembleRecognizer
        return EnsembleRecognizer
    raise AttributeError(name)


__all__ = ["CalligrapherRecognizer", "EnsembleRecognizer"]
