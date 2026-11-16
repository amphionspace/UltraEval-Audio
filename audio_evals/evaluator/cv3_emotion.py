import os
from typing import Dict

from audio_evals.evaluator.base import Evaluator


class CV3EmotionEval(Evaluator):
    """
    Emotion evaluation for CV3-Eval benchmark.

    This evaluator uses emotion2vec_plus_large to recognize emotions from
    generated speech and compares with the reference emotion label.

    The reference emotion is expected to be provided in the kwargs (e.g., from the
    utterance ID format: {emotion}_{level}_... in CV3-Eval dataset).

    Args:
        model_name: Name of the emotion recognition model.
                   Default: 'emotion2vec_plus_large'
    """

    def __init__(
        self,
        model_name: str = "emotion2vec_plus_large",
    ):
        from audio_evals.registry import registry

        self.model = registry.get_model(model_name)

    def _eval(self, pred, label, **kwargs) -> Dict[str, any]:
        """
        Evaluate emotion accuracy.

        Args:
            pred: Path to the generated audio file
            label: Reference label (may contain emotion info or use ref_emotion from kwargs)
            **kwargs: Additional arguments, may contain:
                - ref_emotion: Reference emotion label
                - utterance_id: Utterance ID in format {emotion}_{level}_...

        Returns:
            Dict containing:
                - emotion_pred: Predicted emotion
                - emotion_ref: Reference emotion
                - emotion_score: Confidence score from model
                - emotion_match: 1 if match, 0 otherwise
        """
        pred = str(pred)
        assert os.path.exists(pred), f"Prediction file {pred} does not exist"

        # Get reference emotion from kwargs
        labels = [
            "angry",
            "disgusted",
            "fearful",
            "happy",
            "neutral",
            "other",
            "sad",
            "surprised",
            "unk",
        ]
        # The dataset ref_col is the utterance id, e.g. emotion_zeroshot_happy_high_uttid_1; the
        # official eval_emotation.py takes the emotion from that id. (The loop variable used to
        # shadow `label`, so every row was scored against "angry".)
        tokens = str(kwargs.get("ref_emotion") or label).split("_")
        ref_emotion = next((name for name in labels if name in tokens), None)
        if ref_emotion is None:
            raise ValueError(f"No emotion label in reference id {label!r}")

        # Run emotion recognition on generated audio
        emotion_pred, emotion_score = self.model.inference({"audio": pred})
        emotion_pred = emotion_pred.lower()

        # Check if prediction matches reference
        emotion_match = 1 if emotion_pred == ref_emotion else 0

        return {
            "emotion_pred": emotion_pred,
            "emotion_ref": ref_emotion,
            "emotion_score": emotion_score,
            "emotion_match": emotion_match,
            ref_emotion: emotion_match,
            "pred": pred,
            "ref": label,
        }
