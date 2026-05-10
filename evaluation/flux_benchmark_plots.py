"""
Static chart generation for FLUXSynID face benchmark result folders.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


class BenchmarkPlotError(ValueError):
    """Raised when benchmark chart inputs are missing or malformed."""


@dataclass(frozen=True)
class ModelSummary:
    model_type: str
    model_label: str
    threshold: float
    identities_tested: int
    genuine_trials: int
    genuine_passed: int
    genuine_failed: int
    impostor_trials: int
    impostor_false_accepts: int
    frr: float
    far: float
    output_dir: Path

    @property
    def wrong_identity_rejected(self) -> int:
        return self.impostor_trials - self.impostor_false_accepts

    @property
    def accepted_correct_pct(self) -> float:
        return 1.0 - self.frr

    @property
    def rejected_wrong_pct(self) -> float:
        return 1.0 - self.far


@dataclass(frozen=True)
class TrialScores:
    genuine: np.ndarray
    impostor: np.ndarray


@dataclass(frozen=True)
class ChartArtifacts:
    output_dir: Path
    executive_summary: Path
    buyer_summary_table: Path
    buyer_summary_table_tr: Path
    charts: Dict[str, Path]


def generate_benchmark_chart_report(
    run_dir: Path,
    *,
    output_dir: Optional[Path] = None,
    title: Optional[str] = None,
    top_models: Optional[Sequence[str]] = None,
    image_format: str = "png",
    language: str = "en",
) -> ChartArtifacts:
    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir else run_dir / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_format = image_format.lower().lstrip(".")
    if image_format not in {"png", "svg"}:
        raise BenchmarkPlotError("--format must be png or svg.")
    language = language.lower()
    if language not in {"en", "tr"}:
        raise BenchmarkPlotError("--language must be en or tr.")

    summaries = load_model_summaries(run_dir)
    if not summaries:
        raise BenchmarkPlotError(f"No completed model rows found in {run_dir / 'comparison.csv'}.")
    selected_for_overlay = select_overlay_models(summaries, top_models)
    scores = {summary.model_type: load_trial_scores(summary) for summary in summaries}
    sweep_rows = {
        summary.model_type: load_threshold_sweep(summary)
        for summary in summaries
    }
    report_title = title or _text("default_title", language)
    charts: Dict[str, Path] = {}

    charts["far_frr_comparison"] = output_dir / f"far_frr_comparison.{image_format}"
    plot_far_frr_comparison(summaries, charts["far_frr_comparison"], report_title, language)

    charts["false_accepts_false_rejects"] = output_dir / f"false_accepts_false_rejects.{image_format}"
    plot_error_counts(summaries, charts["false_accepts_false_rejects"], report_title, language)

    for summary in summaries:
        key = f"score_distribution_{summary.model_type}"
        charts[key] = output_dir / f"{key}.{image_format}"
        plot_score_distribution(
            summary,
            scores[summary.model_type],
            charts[key],
            report_title,
            language,
        )
        if sweep_rows[summary.model_type]:
            sweep_key = f"threshold_sweep_{summary.model_type}"
            charts[sweep_key] = output_dir / f"{sweep_key}.{image_format}"
            plot_threshold_sweep(
                summary,
                sweep_rows[summary.model_type],
                charts[sweep_key],
                report_title,
                language,
            )

    charts["score_distribution_overlay"] = output_dir / f"score_distribution_overlay.{image_format}"
    plot_score_overlay(
        selected_for_overlay,
        scores,
        charts["score_distribution_overlay"],
        report_title,
        language,
    )

    buyer_summary_table = output_dir / "buyer_summary_table.csv"
    write_buyer_summary_table(buyer_summary_table, summaries)
    buyer_summary_table_tr = output_dir / "buyer_summary_table_tr.csv"
    write_buyer_summary_table_tr(buyer_summary_table_tr, summaries)

    executive_summary = output_dir / "executive_summary.md"
    executive_summary.write_text(
        render_executive_summary(
            run_dir,
            output_dir,
            summaries,
            selected_for_overlay,
            charts,
            buyer_summary_table,
            buyer_summary_table_tr,
            report_title,
            language,
        ),
        encoding="utf-8",
    )
    return ChartArtifacts(
        output_dir=output_dir,
        executive_summary=executive_summary,
        buyer_summary_table=buyer_summary_table,
        buyer_summary_table_tr=buyer_summary_table_tr,
        charts=charts,
    )


def load_model_summaries(run_dir: Path) -> List[ModelSummary]:
    comparison_path = Path(run_dir) / "comparison.csv"
    if not comparison_path.exists():
        raise BenchmarkPlotError(f"Missing required benchmark file: {comparison_path}")
    rows = _read_csv(comparison_path)
    summaries: List[ModelSummary] = []
    for row in rows:
        if row.get("status") != "completed":
            continue
        summaries.append(ModelSummary(
            model_type=row["model_type"],
            model_label=row.get("model_label") or row["model_type"],
            threshold=_to_float(row["threshold"], "threshold"),
            identities_tested=_to_int(row["identities_tested"], "identities_tested"),
            genuine_trials=_to_int(row["genuine_trials"], "genuine_trials"),
            genuine_passed=_to_int(row["genuine_passed"], "genuine_passed"),
            genuine_failed=_to_int(row["genuine_failed"], "genuine_failed"),
            impostor_trials=_to_int(row["impostor_trials"], "impostor_trials"),
            impostor_false_accepts=_to_int(
                row["impostor_false_accepts"],
                "impostor_false_accepts",
            ),
            frr=_to_float(row["frr"], "frr"),
            far=_to_float(row["far"], "far"),
            output_dir=Path(row["output_dir"]),
        ))
    return summaries


def load_trial_scores(summary: ModelSummary) -> TrialScores:
    path = summary.output_dir / "results.csv"
    if not path.exists():
        raise BenchmarkPlotError(f"Missing per-model results file: {path}")
    genuine: List[float] = []
    impostor: List[float] = []
    for row in _read_csv(path):
        score = _to_float(row["score"], "score")
        if row["trial_type"] == "genuine":
            genuine.append(score)
        elif row["trial_type"] == "impostor":
            impostor.append(score)
    if not genuine or not impostor:
        raise BenchmarkPlotError(f"Expected genuine and impostor trials in {path}")
    return TrialScores(
        genuine=np.asarray(genuine, dtype=np.float64),
        impostor=np.asarray(impostor, dtype=np.float64),
    )


def load_threshold_sweep(summary: ModelSummary) -> List[dict]:
    path = summary.output_dir / "threshold_sweep.csv"
    if not path.exists():
        return []
    rows = _read_csv(path)
    return [
        {
            "threshold": _to_float(row["threshold"], "threshold"),
            "far": _to_float(row["far"], "far"),
            "frr": _to_float(row["frr"], "frr"),
        }
        for row in rows
        if row.get("threshold")
    ]


def select_best_model(summaries: Sequence[ModelSummary]) -> ModelSummary:
    if not summaries:
        raise BenchmarkPlotError("Cannot select best model from an empty result set.")
    return sorted(summaries, key=lambda item: (item.far, item.frr, item.model_label))[0]


def select_overlay_models(
    summaries: Sequence[ModelSummary],
    requested: Optional[Sequence[str]] = None,
) -> List[ModelSummary]:
    by_type = {summary.model_type: summary for summary in summaries}
    if requested:
        missing = [model for model in requested if model not in by_type]
        if missing:
            raise BenchmarkPlotError(f"Unknown --top-models value(s): {', '.join(missing)}")
        return [by_type[model] for model in requested]
    return sorted(summaries, key=lambda item: (item.far, item.frr, item.model_label))[:2]


def plot_far_frr_comparison(
    summaries: Sequence[ModelSummary],
    path: Path,
    title: str,
    language: str,
) -> None:
    best = select_best_model(summaries)
    labels = [_short_label(item, language) for item in summaries]
    x = np.arange(len(summaries))
    width = 0.34
    far = [item.far * 100 for item in summaries]
    frr = [item.frr * 100 for item in summaries]

    fig, ax = plt.subplots(figsize=(11, 6.2))
    far_bars = ax.bar(x - width / 2, far, width, label=_text("far_label", language), color="#E4002B")
    frr_bars = ax.bar(x + width / 2, frr, width, label=_text("frr_label", language), color="#AD841F")
    ax.set_title(f"{title}\n{_text('far_frr_title', language)}", fontsize=16, weight="bold")
    ax.set_ylabel(_text("rate_axis", language), fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([
        f"{label}\n★ {_text('best', language)}" if item.model_type == best.model_type else label
        for label, item in zip(labels, summaries)
    ], fontsize=10)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    _label_bars(ax, far_bars, suffix="%")
    _label_bars(ax, frr_bars, suffix="%")
    _save_fig(fig, path)


def plot_error_counts(
    summaries: Sequence[ModelSummary],
    path: Path,
    title: str,
    language: str,
) -> None:
    labels = [_short_label(item, language) for item in summaries]
    x = np.arange(len(summaries))
    width = 0.34
    false_accepts = [item.impostor_false_accepts for item in summaries]
    false_rejects = [item.genuine_failed for item in summaries]

    fig, ax = plt.subplots(figsize=(11, 6.2))
    accept_bars = ax.bar(x - width / 2, false_accepts, width, label=_text("wrong_accepted", language), color="#E4002B")
    reject_bars = ax.bar(x + width / 2, false_rejects, width, label=_text("correct_rejected", language), color="#003087")
    ax.set_title(f"{title}\n{_text('raw_error_title', language)}", fontsize=16, weight="bold")
    ax.set_ylabel(_text("count_axis", language), fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    _label_bars(ax, accept_bars, decimals=0)
    _label_bars(ax, reject_bars, decimals=0)
    _save_fig(fig, path)


def plot_score_distribution(
    summary: ModelSummary,
    scores: TrialScores,
    path: Path,
    title: str,
    language: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6))
    bins = np.linspace(0, 1, 41)
    ax.hist(
        scores.genuine,
        bins=bins,
        density=True,
        alpha=0.60,
        label=_text("correct_scores", language),
        color="#003087",
    )
    ax.hist(
        scores.impostor,
        bins=bins,
        density=True,
        alpha=0.55,
        label=_text("wrong_scores", language),
        color="#E4002B",
    )
    ax.axvline(
        summary.threshold,
        color="#182033",
        linestyle="--",
        linewidth=2,
        label=f"{_text('decision_threshold', language)} {summary.threshold:.3f}",
    )
    ax.set_xlim(0, 1)
    ax.set_title(
        f"{title}\n{_display_model_label(summary, language)} {_text('score_distribution_title', language)}",
        fontsize=16,
        weight="bold",
    )
    ax.set_xlabel(_text("match_score_axis", language), fontsize=12)
    ax.set_ylabel(_text("density_axis", language), fontsize=12)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.20)
    _save_fig(fig, path)


def plot_score_overlay(
    summaries: Sequence[ModelSummary],
    score_map: Dict[str, TrialScores],
    path: Path,
    title: str,
    language: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharex=True)
    bins = np.linspace(0, 1, 51)
    colors = ["#003087", "#E4002B", "#AD841F", "#667085"]
    for index, summary in enumerate(summaries):
        color = colors[index % len(colors)]
        scores = score_map[summary.model_type]
        axes[0].hist(
            scores.genuine,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.3,
            color=color,
            label=_display_model_label(summary, language),
        )
        axes[1].hist(
            scores.impostor,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.3,
            color=color,
            label=_display_model_label(summary, language),
        )
        axes[0].axvline(summary.threshold, color=color, linestyle="--", linewidth=1.4, alpha=0.75)
        axes[1].axvline(summary.threshold, color=color, linestyle="--", linewidth=1.4, alpha=0.75)

    axes[0].set_title(_text("correct_scores", language), fontsize=13, weight="bold")
    axes[1].set_title(_text("wrong_scores", language), fontsize=13, weight="bold")
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_xlabel(_text("match_score_short", language), fontsize=11)
        ax.grid(axis="y", alpha=0.20)
        ax.legend()
    axes[0].set_ylabel(_text("density_axis", language), fontsize=11)
    fig.suptitle(f"{title}\n{_text('top_model_separation', language)}", fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    _save_fig(fig, path)


def plot_threshold_sweep(
    summary: ModelSummary,
    rows: Sequence[dict],
    path: Path,
    title: str,
    language: str,
) -> None:
    thresholds = [row["threshold"] for row in rows]
    far = [row["far"] * 100 for row in rows]
    frr = [row["frr"] * 100 for row in rows]
    fig, ax = plt.subplots(figsize=(10.5, 6))
    ax.plot(thresholds, far, marker="o", linewidth=2.2, label=_text("far_sweep_label", language), color="#E4002B")
    ax.plot(thresholds, frr, marker="o", linewidth=2.2, label=_text("frr_sweep_label", language), color="#003087")
    ax.axvline(summary.threshold, color="#182033", linestyle="--", linewidth=2, label=_text("active_threshold", language))
    ax.set_title(
        f"{title}\n{_display_model_label(summary, language)} {_text('threshold_sweep_title', language)}",
        fontsize=16,
        weight="bold",
    )
    ax.set_xlabel(_text("decision_threshold_axis", language), fontsize=12)
    ax.set_ylabel(_text("rate_axis", language), fontsize=12)
    ax.set_xlim(min(thresholds), max(thresholds))
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)
    _save_fig(fig, path)


def write_buyer_summary_table(path: Path, summaries: Sequence[ModelSummary]) -> None:
    fields = [
        "model_type",
        "model_label",
        "threshold",
        "correct_students_accepted",
        "correct_students_rejected",
        "wrong_identities_rejected",
        "wrong_identities_accepted",
        "frr_percent",
        "far_percent",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({
                "model_type": summary.model_type,
                "model_label": summary.model_label,
                "threshold": _fmt(summary.threshold, 6),
                "correct_students_accepted": summary.genuine_passed,
                "correct_students_rejected": summary.genuine_failed,
                "wrong_identities_rejected": summary.wrong_identity_rejected,
                "wrong_identities_accepted": summary.impostor_false_accepts,
                "frr_percent": _fmt(summary.frr * 100),
                "far_percent": _fmt(summary.far * 100),
            })


def write_buyer_summary_table_tr(path: Path, summaries: Sequence[ModelSummary]) -> None:
    fields = [
        "model_tipi",
        "model_adi",
        "esik_degeri",
        "dogru_ogrenci_kabul",
        "dogru_ogrenci_ret",
        "yanlis_kimlik_ret",
        "yanlis_kimlik_kabul",
        "frr_yuzde",
        "far_yuzde",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({
                "model_tipi": summary.model_type,
                "model_adi": _turkish_model_label(summary),
                "esik_degeri": _fmt(summary.threshold, 6),
                "dogru_ogrenci_kabul": summary.genuine_passed,
                "dogru_ogrenci_ret": summary.genuine_failed,
                "yanlis_kimlik_ret": summary.wrong_identity_rejected,
                "yanlis_kimlik_kabul": summary.impostor_false_accepts,
                "frr_yuzde": _fmt(summary.frr * 100),
                "far_yuzde": _fmt(summary.far * 100),
            })


def render_executive_summary(
    run_dir: Path,
    output_dir: Path,
    summaries: Sequence[ModelSummary],
    overlay_models: Sequence[ModelSummary],
    charts: Dict[str, Path],
    buyer_summary_table: Path,
    buyer_summary_table_tr: Path,
    title: str,
    language: str,
) -> str:
    best = select_best_model(summaries)
    compared = ", ".join(_display_model_label(summary, language) for summary in summaries)
    overlay = ", ".join(_display_model_label(summary, language) for summary in overlay_models)
    if language == "tr":
        best_label = _display_model_label(best, language)
        lines = [
            f"# {title}",
            "",
            "FaceVerify FLUXSynID benchmark çıktılarından oluşturulmuş statik görsel rapor.",
            "",
            "## Yönetici Özeti",
            "",
            f"- Benchmark klasörü: `{run_dir}`",
            f"- Karşılaştırılan modeller: {compared}",
            f"- En düşük FAR, sonra en düşük FRR ölçütüne göre en iyi model: **{best_label}**",
            f"- {best_label}, doğru öğrencilerin **%{_fmt(best.accepted_correct_pct * 100)}** oranını kabul etti "
            f"ve yanlış kimlik denemelerinin **%{_fmt(best.rejected_wrong_pct * 100)}** oranını reddetti.",
            f"- `{_fmt(best.threshold, 6)}` eşik değerinde `{best.genuine_trials}` doğru öğrenci denemesinde "
            f"`{best.genuine_failed}` yanlış ret ve `{best.impostor_trials}` yanlış kimlik denemesinde "
            f"`{best.impostor_false_accepts}` yanlış kabul oluştu.",
            "",
            "## Grafikler",
            "",
            f"- FAR/FRR karşılaştırması: [{charts['far_frr_comparison'].name}]({charts['far_frr_comparison'].name})",
            f"- Ham hata sayıları: [{charts['false_accepts_false_rejects'].name}]({charts['false_accepts_false_rejects'].name})",
            f"- {overlay} skor dağılımı: [{charts['score_distribution_overlay'].name}]({charts['score_distribution_overlay'].name})",
            f"- Türkçe özet tablo: [{buyer_summary_table_tr.name}]({buyer_summary_table_tr.name})",
            f"- İngilizce özet tablo: [{buyer_summary_table.name}]({buyer_summary_table.name})",
            "",
            "## Model Bazlı Ek Grafikler",
            "",
        ]
        for summary in summaries:
            label = _display_model_label(summary, language)
            score_key = f"score_distribution_{summary.model_type}"
            sweep_key = f"threshold_sweep_{summary.model_type}"
            lines.append(f"- {label}: [{charts[score_key].name}]({charts[score_key].name})")
            if sweep_key in charts:
                lines.append(f"- {label} eşik analizi: [{charts[sweep_key].name}]({charts[sweep_key].name})")
        lines.extend([
            "",
            "## Not",
            "",
            "Bu grafikler yalnızca raporlama içindir. Model checkpoint dosyalarını, çalışma zamanı eşiklerini veya benchmark CSV dosyalarını değiştirmez.",
            "",
            f"Üretilen çıktılar: `{output_dir}`.",
        ])
        return "\n".join(lines) + "\n"

    lines = [
        f"# {title}",
        "",
        "Static visual report generated from FaceVerify FLUXSynID benchmark artifacts.",
        "",
        "## Buyer Summary",
        "",
        f"- Benchmark folder: `{run_dir}`",
        f"- Models compared: {compared}",
        f"- Best model by lowest FAR then FRR: **{best.model_label}**",
        f"- {best.model_label} accepted **{_fmt(best.accepted_correct_pct * 100)}%** of correct students "
        f"and rejected **{_fmt(best.rejected_wrong_pct * 100)}%** of wrong-identity attempts.",
        f"- At threshold `{_fmt(best.threshold, 6)}`, it had `{best.genuine_failed}` false rejections "
        f"from `{best.genuine_trials}` correct-student trials and `{best.impostor_false_accepts}` "
        f"false accepts from `{best.impostor_trials}` wrong-identity trials.",
        "",
        "## Charts",
        "",
        f"- FAR/FRR comparison: [{charts['far_frr_comparison'].name}]({charts['far_frr_comparison'].name})",
        f"- Raw error counts: [{charts['false_accepts_false_rejects'].name}]({charts['false_accepts_false_rejects'].name})",
        f"- Score overlay for {overlay}: [{charts['score_distribution_overlay'].name}]({charts['score_distribution_overlay'].name})",
        f"- Buyer summary table: [{buyer_summary_table.name}]({buyer_summary_table.name})",
        f"- Turkish buyer summary table: [{buyer_summary_table_tr.name}]({buyer_summary_table_tr.name})",
        "",
        "## Per-Model Diagnostics",
        "",
    ]
    for summary in summaries:
        score_key = f"score_distribution_{summary.model_type}"
        sweep_key = f"threshold_sweep_{summary.model_type}"
        lines.append(f"- {summary.model_label}: [{charts[score_key].name}]({charts[score_key].name})")
        if sweep_key in charts:
            lines.append(f"- {summary.model_label} threshold sweep: [{charts[sweep_key].name}]({charts[sweep_key].name})")
    lines.extend([
        "",
        "## Note",
        "",
        "These charts are report-only. They do not modify model checkpoints, runtime thresholds, or benchmark CSVs.",
        "",
        f"Generated artifacts are in `{output_dir}`.",
    ])
    return "\n".join(lines) + "\n"


def _read_csv(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(value: str, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise BenchmarkPlotError(f"Invalid numeric value for {field}: {value!r}") from exc


def _to_int(value: str, field: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError) as exc:
        raise BenchmarkPlotError(f"Invalid integer value for {field}: {value!r}") from exc


def _short_label(summary: ModelSummary, language: str = "en") -> str:
    replacements = {
        "en": {
            "FaceNet Contrastive Proto": "Contrastive Proto",
            "Hybrid FaceNet": "Hybrid",
            "FaceNet Proto": "FaceNet Proto",
        },
        "tr": {
            "FaceNet Contrastive Proto": "Kontrastif Proto",
            "Hybrid FaceNet": "Hibrit",
            "FaceNet Proto": "FaceNet Proto",
        },
    }
    return replacements.get(language, replacements["en"]).get(
        summary.model_label,
        _display_model_label(summary, language),
    )


def _display_model_label(summary: ModelSummary, language: str = "en") -> str:
    return _turkish_model_label(summary) if language == "tr" else summary.model_label


def _turkish_model_label(summary: ModelSummary) -> str:
    labels = {
        "Hybrid FaceNet": "Hibrit FaceNet",
        "FaceNet Contrastive Proto": "FaceNet Kontrastif Proto",
        "FaceNet Proto": "FaceNet Proto",
        "Siamese": "Siamese",
        "Prototypical": "Prototipik",
    }
    return labels.get(summary.model_label, summary.model_label)


def _text(key: str, language: str = "en") -> str:
    labels = {
        "en": {
            "active_threshold": "Active threshold",
            "best": "Best",
            "correct_rejected": "Correct students rejected",
            "correct_scores": "Correct student scores",
            "count_axis": "Count",
            "decision_threshold": "Decision threshold",
            "decision_threshold_axis": "Decision threshold",
            "default_title": "FLUXSynID Face Model Benchmark",
            "density_axis": "Density",
            "far_frr_title": "False Acceptance vs False Rejection",
            "far_label": "Wrong identity accepted (FAR)",
            "far_sweep_label": "FAR: wrong identity accepted",
            "frr_label": "Correct student rejected (FRR)",
            "frr_sweep_label": "FRR: correct student rejected",
            "match_score_axis": "Match score (higher means stronger match)",
            "match_score_short": "Match score",
            "rate_axis": "Rate (%)",
            "raw_error_title": "Raw Error Counts",
            "score_distribution_title": "Score Distribution",
            "threshold_sweep_title": "Threshold Sweep",
            "top_model_separation": "Top Model Score Separation",
            "wrong_accepted": "Wrong identities accepted",
            "wrong_scores": "Wrong identity scores",
        },
        "tr": {
            "active_threshold": "Aktif eşik",
            "best": "En iyi",
            "correct_rejected": "Reddedilen doğru öğrenciler",
            "correct_scores": "Doğru öğrenci skorları",
            "count_axis": "Sayı",
            "decision_threshold": "Karar eşiği",
            "decision_threshold_axis": "Karar eşiği",
            "default_title": "FLUXSynID Yüz Modeli Karşılaştırması",
            "density_axis": "Yoğunluk",
            "far_frr_title": "Yanlış Kabul ve Yanlış Ret Karşılaştırması",
            "far_label": "Kabul edilen yanlış kimlik (FAR)",
            "far_sweep_label": "FAR: kabul edilen yanlış kimlik",
            "frr_label": "Reddedilen doğru öğrenci (FRR)",
            "frr_sweep_label": "FRR: reddedilen doğru öğrenci",
            "match_score_axis": "Eşleşme skoru (yüksek değer daha güçlü eşleşme)",
            "match_score_short": "Eşleşme skoru",
            "rate_axis": "Oran (%)",
            "raw_error_title": "Ham Hata Sayıları",
            "score_distribution_title": "Skor Dağılımı",
            "threshold_sweep_title": "Eşik Analizi",
            "top_model_separation": "En İyi Modellerin Skor Ayrımı",
            "wrong_accepted": "Kabul edilen yanlış kimlikler",
            "wrong_scores": "Yanlış kimlik skorları",
        },
    }
    return labels.get(language, labels["en"]).get(key, labels["en"][key])


def _label_bars(ax, bars, suffix: str = "", decimals: int = 2) -> None:
    for bar in bars:
        height = bar.get_height()
        if decimals == 0:
            label = f"{int(round(height))}{suffix}"
        else:
            label = f"{height:.{decimals}f}{suffix}"
        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def _fmt(value: float, decimals: int = 2) -> str:
    return f"{float(value):.{decimals}f}".rstrip("0").rstrip(".")


def _save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
