import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_flux_controls_and_previews_are_wired_in_static_ui():
    index = (PROJECT_ROOT / "ui" / "index.html").read_text(encoding="utf-8")
    app = (PROJECT_ROOT / "ui" / "js" / "app.js").read_text(encoding="utf-8")
    i18n = (PROJECT_ROOT / "ui" / "js" / "i18n.js").read_text(encoding="utf-8")
    css = (PROJECT_ROOT / "ui" / "css" / "style.css").read_text(encoding="utf-8")
    dark_css = (PROJECT_ROOT / "ui" / "css" / "dark-theme.css").read_text(encoding="utf-8")

    for element_id in [
        "flux-preupload-btn",
        "flux-export-btn",
        "flux-download-zip",
        "flux-test-output",
        "flux-test-count",
        "flux-status",
        "verify-preloaded-btn",
        "simulation-preloaded-btn",
        "preloaded-stage-note",
        "simulation-preloaded-stage-note",
        "simulation-student-avatar",
        "gate-student-avatar",
        "gate-download-test-image",
    ]:
        assert f'id="{element_id}"' in index

    assert "async function preuploadFlux" in app
    assert "async function exportFluxTestSet" in app
    assert "function stagePreloadedSelfie" in app
    assert "function confirmVerification" in app
    assert "function setVerificationInFlight" in app
    assert "studentFluxTestImageUrl" in app
    assert '<html lang="tr">' in index
    assert '<script src="/js/i18n.js"></script>' in index
    assert index.index('/js/i18n.js') < index.index('/js/api.js')
    assert "Load Preloaded Selfie" in index
    assert "Hazır Selfieyi Yükle" in i18n
    assert "FaceVerifyI18n" in i18n
    assert '<option value="facenet_proto">FaceNet Proto</option>' in index
    assert (
        '<option value="facenet_contrastive_proto" selected>'
        'FaceNet Contrastive Proto</option>'
    ) in index
    assert '<option value="hybrid" selected>Hybrid FaceNet</option>' not in index
    assert "facenet_proto: 'FaceNet Proto'" in app
    assert "facenet_contrastive_proto: 'FaceNet Contrastive Proto'" in app
    assert "function applySelectedModelDefaultThreshold" in app
    assert "function formatThresholdInput" in app
    assert "facenet_proto: { threshold: 0.47 }" in app
    assert "facenet_contrastive_proto: { threshold: 0.800884 }" in app
    assert "Hybrid FaceNet, FaceNet Proto ve FaceNet Contrastive Proto" in i18n
    assert "Use Preloaded Selfie" not in index
    assert "studentNameCell(row)" in app
    assert ".student-avatar" in css


def test_simulation_dashboard_enhancement_ui_is_wired():
    index = (PROJECT_ROOT / "ui" / "index.html").read_text(encoding="utf-8")
    app = (PROJECT_ROOT / "ui" / "js" / "app.js").read_text(encoding="utf-8")
    i18n = (PROJECT_ROOT / "ui" / "js" / "i18n.js").read_text(encoding="utf-8")
    css = (PROJECT_ROOT / "ui" / "css" / "style.css").read_text(encoding="utf-8")
    dark_css = (PROJECT_ROOT / "ui" / "css" / "dark-theme.css").read_text(encoding="utf-8")

    for element_id in [
        "gate-step-title",
        "gate-status-pill",
        "gate-step-pill",
        "simulation-gate-title",
        "simulation-gate-status-pill",
        "simulation-gate-step-pill",
        "simulation-kpi-attempts-count",
        "simulation-kpi-verified-spark",
        "review-kpi-review-count",
        "review-kpi-no-enrollment-spark",
        "simulation-feed-grid",
        "simulation-evidence-score",
        "simulation-evidence-rail",
        "simulation-audit-rail",
        "review-audit-rail",
        "sidebar-clock-time",
        "sidebar-clock-date",
    ]:
        assert f'id="{element_id}"' in index
    assert 'id="simulation-feed-table"' not in index
    assert 'class="verification-workbench"' in index
    assert index.index('class="attempt-review-card"') < index.index('class="student-verification-panel"')

    assert 'data-theme-option="light"' in index
    assert 'data-theme-option="dark"' in index
    assert "faceverifyTheme" in index
    assert index.index("faceverifyTheme") < index.index('/css/style.css')
    assert 'id="theme-dark-stylesheet"' in index

    for function_name in [
        "function renderKpiDeck",
        "function renderAuditRail",
        "function renderGateHeader",
        "function updateSidebarClock",
        "function initTheme",
        "function setTheme",
        "function renderThemeToggle",
    ]:
        assert function_name in app

    assert "THEME_STORAGE_KEY = 'faceverifyTheme'" in app
    assert "document.documentElement.dataset.theme" in app

    for selector in [
        ".dashboard-kpi-grid",
        ".verification-workbench",
        ".student-verification-panel",
        ".student-feed-grid",
        ".student-feed-card",
        ".feed-photo-frame",
        ".simulation-evidence-panel",
        ".simulation-evidence-rail",
        ".gate-card-header",
        ".verification-audit-rail",
        ".sidebar-status-card",
    ]:
        assert selector in css

    assert 'html[data-theme="dark"]' in dark_css
    assert 'html[data-theme="light"] .theme-toggle' in dark_css
    assert '--fallback: #8b5cf6' in dark_css
    assert 'html[data-theme="light"] .simulation-layout' in dark_css
    assert 'html[data-theme="light"] .simulation-divider' in dark_css
    assert 'html[data-theme="light"] .kpi-card:hover' in dark_css
    assert 'html[data-theme="light"] .camera-preview::before' in dark_css
    assert 'html[data-theme="light"] .verification-audit-rail' in dark_css
    assert 'html[data-theme="light"] .student-feed-card' in dark_css
    assert 'html[data-theme="light"] .student-verification-panel' in dark_css
    assert 'html[data-theme="light"] .simulation-evidence-panel' in dark_css
    assert 'html[data-theme="light"] #operation-model-evidence' in dark_css
    assert 'simulationSubmittedPhoto' in app
    assert 'feedImageSlot' in app
    assert 'function renderSimulationEvidence' in app

    for localization_symbol in [
        "function statusLabel",
        "function decisionLabel",
        "function formatDateTime",
        "function apiErrorMessage",
        "MutationObserver",
        "Intl.DateTimeFormat",
    ]:
        assert localization_symbol in i18n

    for turkish_label in [
        "Doğrulandı",
        "Manuel İnceleme",
        "Reddedildi",
        "Sınav Erişimi İçin Doğrula",
        "tr-TR",
    ]:
        assert turkish_label in i18n
