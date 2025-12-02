from __future__ import annotations

import csv
import json
import os
import sys
import threading
import uuid
from dataclasses import dataclass, field
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# Make the backend package importable.
ROOT = Path(__file__).resolve().parent
BACKEND_SRC = ROOT.parent / "prompt2song_service" / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from prompt2song import service  # noqa: E402
from prompt2song.rlhf import (  # noqa: E402
    FEATURE_KEYS,
    PreferenceVectorLogger,
    extract_audio_features,
    rerank_candidates,
)


def _spotify_id(song: dict[str, Any]) -> str | None:
    return (
        song.get("song_id")
        or song.get("id")
        or song.get("track_id")
        or song.get("trackId")
    )


def _clean_value(value: Any) -> Any:
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _json_default(obj: Any) -> Any:
    cleaned = _clean_value(obj)
    if cleaned is obj:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    return cleaned


def _song_display(song: dict[str, Any]) -> str:
    title = song.get("name") or "Unknown title"
    artists = song.get("artists") or "Unknown artist"
    return f"\"{title}\" – {artists}"


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not path or not path.exists():
        return rows
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


@dataclass
class SessionState:
    prompt: str
    top_k: int | None
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    answered: int = 0
    logs: list[str] = field(default_factory=list)
    classification: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = field(default_factory=list)
    feature_vectors: list[np.ndarray] = field(default_factory=list)
    preference_vector: np.ndarray | None = None
    asked_pairs: set[int] = field(default_factory=set)
    current_pair: tuple[int, int] | None = None
    phase1_csv: Path | None = None
    final_csv: Path | None = None
    completed: bool = False
    weight_log_path: Path | None = None
    weight_logger: PreferenceVectorLogger | None = None

    def __post_init__(self) -> None:
        rlhf_cfg = service.settings.rlhf
        retrieval_cfg = service.settings.retrieval
        self.pool_k = self.top_k or retrieval_cfg.top_k
        self.final_top_k = self.top_k if self.top_k is not None else (
            rlhf_cfg.final_top_k or retrieval_cfg.top_k
        )
        self.num_questions = rlhf_cfg.num_rlhf_questions
        self.learning_rate = rlhf_cfg.learning_rate
        self.preference_weight = rlhf_cfg.preference_weight
        self.phase1_name, self.phase2_name = service.resolve_output_filenames(None)

        self.logs.append(f"Prompt: {self.prompt}")
        self.classification = service.classify_prompt(self.prompt)
        if self.classification:
            label = self.classification.get("label")
            score = self.classification.get("score")
            if label is not None and score is not None:
                self.logs.append(f"Prompt classification: {label} ({score:.2f} confidence)")
            elif label:
                self.logs.append(f"Prompt classification: {label}")
            probs = self.classification.get("probabilities") if isinstance(self.classification, dict) else None
            if probs:
                ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
                prob_str = ", ".join(f"{name}: {val:.2f}" for name, val in ranked)
                self.logs.append(f"Class probabilities: {prob_str}")
        self.logs.append(f"Retrieving top {self.pool_k} candidates...")

        self.candidates = service.recommender.recommend(self.prompt, self.pool_k)
        self.feature_vectors = [extract_audio_features(c) for c in self.candidates]
        if self.feature_vectors:
            self.preference_vector = np.zeros(self.feature_vectors[0].shape, dtype=float)
        else:
            self.preference_vector = np.zeros(0, dtype=float)
        if self.num_questions > 0 and self.feature_vectors:
            log_root = service.settings.paths.output_dir / "rlhf_logs"
            session_dir = log_root / f"session_{self.session_id}"
            self.weight_log_path = session_dir / "rlhf_weights.csv"
            self.weight_logger = PreferenceVectorLogger(
                self.weight_log_path, session_id=self.session_id, feature_keys=FEATURE_KEYS
            )
            self.weight_logger.log(0, self.preference_vector)

        self.logs.append(f"Candidate pool ready ({len(self.candidates)} songs)")
        if self.candidates:
            self.phase1_csv = service.exporter.to_csv(self.candidates, filename=self.phase1_name)
            self.logs.append(f"Wrote Phase 1 recommendations to {self.phase1_csv}")
        self.logs.append(
            f"RLHF configured for {self.num_questions} questions, learning_rate={self.learning_rate}, "
            f"preference_weight={self.preference_weight}, final_top_k={self.final_top_k}"
        )
        if self.weight_log_path:
            self.logs.append(f"Logging preference vectors to {self.weight_log_path}")

    def _rank_indices(self) -> list[int]:
        scores: list[tuple[int, float]] = []
        for idx, (cand, feats) in enumerate(zip(self.candidates, self.feature_vectors)):
            base = cand.get("score", 0.0)
            pref = float(self.preference_vector @ feats) if self.preference_vector.size else 0.0
            final = base + self.preference_weight * pref
            scores.append((idx, final))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in scores]

    def _max_pairs(self) -> int:
        n = len(self.candidates)
        return n * (n - 1) // 2

    def next_question(self) -> dict[str, Any] | None:
        if self.completed or self.num_questions <= 0:
            return None
        if self.answered >= self.num_questions:
            return None
        ranking = self._rank_indices()
        available = [idx for idx in ranking if idx not in self.asked_pairs]
        if len(available) < 2:
            return None

        a_idx, b_idx = available[0], available[1]
        self.current_pair = (a_idx, b_idx)
        question_no = self.answered + 1
        self.logs.append(
            f"RLHF question {question_no}/{self.num_questions}: "
            f"A) {_song_display(self.candidates[a_idx])} | "
            f"B) {_song_display(self.candidates[b_idx])}"
        )
        return self._question_payload(a_idx, b_idx, question_no)

    def _question_payload(self, a_idx: int, b_idx: int, question_no: int) -> dict[str, Any]:
        def payload_from_idx(idx: int) -> dict[str, Any]:
            song = self.candidates[idx]
            sid = _spotify_id(song)
            return {
                "name": song.get("name"),
                "artists": song.get("artists"),
                "album": song.get("album_name"),
                "song_id": sid,
                "spotify_embed": f"https://open.spotify.com/embed/track/{sid}" if sid else None,
            }

        return {
            "index": question_no,
            "total": self.num_questions,
            "a": payload_from_idx(a_idx),
            "b": payload_from_idx(b_idx),
        }

    def record_choice(self, choice: str) -> None:
        if self.current_pair is None or not self.feature_vectors:
            return
        a_idx, b_idx = self.current_pair
        a_vec = self.feature_vectors[a_idx]
        b_vec = self.feature_vectors[b_idx]
        answered_before = self.answered

        if choice == "a":
            self.preference_vector = self.preference_vector + self.learning_rate * (a_vec - b_vec)
            self.answered += 1
            self.logs.append(f"User selected A for question {answered_before + 1}")
        elif choice == "b":
            self.preference_vector = self.preference_vector + self.learning_rate * (b_vec - a_vec)
            self.answered += 1
            self.logs.append(f"User selected B for question {answered_before + 1}")
        else:
            self.logs.append(f"User skipped question {answered_before + 1}")

        if self.weight_logger and choice in ("a", "b"):
            self.weight_logger.log(self.answered, self.preference_vector)
        self.asked_pairs.update({a_idx, b_idx})
        self.current_pair = None

    def finalize(self) -> list[dict[str, Any]]:
        if self.completed:
            return getattr(self, "final_recommendations", [])

        if not self.candidates:
            self.final_recommendations = []
            self.completed = True
            return self.final_recommendations

        reranked = rerank_candidates(
            self.candidates,
            self.feature_vectors,
            preference_vector=self.preference_vector,
            preference_weight=self.preference_weight,
            final_top_k=self.final_top_k,
        )
        self.final_recommendations = reranked
        self.final_csv = service.exporter.to_csv(reranked, filename=self.phase2_name)
        self.logs.append(f"Wrote final recommendations to {self.final_csv}")
        self.completed = True
        return self.final_recommendations


SESSIONS: dict[str, SessionState] = {}
SESSIONS_LOCK = threading.Lock()


def _build_response(
    session: SessionState,
    question: dict[str, Any] | None,
    recommendations: Iterable[dict[str, Any]] | None,
) -> dict[str, Any]:
    final_csv_payload = None
    if session.final_csv:
        final_csv_payload = {
            "path": str(session.final_csv),
            "filename": session.final_csv.name,
            "rows": _read_csv_rows(session.final_csv),
        }
    return {
        "sessionId": session.session_id,
        "logs": session.logs,
        "promptClassification": session.classification,
        "question": question,
        "completed": session.completed,
        "recommendations": list(recommendations) if recommendations is not None else None,
        "phase1Csv": str(session.phase1_csv) if session.phase1_csv else None,
        "finalCsv": final_csv_payload,
    }


class FrontendHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        # Keep console output minimal.
        sys.stderr.write("%s\n" % (format % args))

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload, default=_json_default).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _parse_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length > 0 else b""
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/start":
            self._handle_start()
            return
        if self.path == "/api/answer":
            self._handle_answer()
            return
        self.send_error(404, "Unknown endpoint")

    def _handle_start(self) -> None:
        try:
            body = self._parse_json()
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON"}, status=400)
            return

        prompt = (body.get("prompt") or "").strip()
        if not prompt:
            self._send_json({"error": "Prompt is required"}, status=400)
            return
        top_k = body.get("top_k")
        top_k_int = None
        if top_k is not None:
            try:
                top_k_int = int(top_k)
            except (TypeError, ValueError):
                self._send_json({"error": "top_k must be an integer"}, status=400)
                return

        session = SessionState(prompt=prompt, top_k=top_k_int)
        with SESSIONS_LOCK:
            SESSIONS[session.session_id] = session

        question = session.next_question()
        recommendations = None
        if question is None:
            recommendations = session.finalize()
        payload = _build_response(session, question, recommendations)
        self._send_json(payload)

    def _handle_answer(self) -> None:
        try:
            body = self._parse_json()
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON"}, status=400)
            return

        session_id = body.get("sessionId")
        choice = (body.get("choice") or "").lower()
        if not session_id:
            self._send_json({"error": "sessionId is required"}, status=400)
            return

        with SESSIONS_LOCK:
            session = SESSIONS.get(session_id)
        if not session:
            self._send_json({"error": "Unknown session"}, status=404)
            return

        if session.completed:
            payload = _build_response(session, None, session.final_recommendations)
            self._send_json(payload)
            return

        session.record_choice(choice)
        question = session.next_question()
        recommendations = None
        if question is None:
            recommendations = session.finalize()

        payload = _build_response(session, question, recommendations)
        self._send_json(payload)


def run(host: str = "localhost", port: int = 8000) -> None:
    os.chdir(ROOT)
    server = ThreadingHTTPServer((host, port), FrontendHandler)
    print(f"Serving Prompt2Song frontend at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    run()
