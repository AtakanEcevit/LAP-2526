import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _button_attrs_are_wired(attrs: str) -> bool:
    return any(marker in attrs for marker in [" id=", " data-", " disabled"])


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
    assert '<option value="facenet_contrastive_proto">FaceNet Contrastive Proto</option>' in index
    assert index.count(
        '<option value="facenet_contrastive_proto_model5">'
        'FaceNet Contrastive Proto Model 5</option>'
    ) == 3
    assert index.count(
        '<option value="facenet_arcface_triplet_model6" selected>'
        'FaceNet ArcFace Triplet Model 6</option>'
    ) == 3
    assert '<option value="hybrid" selected>Hybrid FaceNet</option>' not in index
    assert "DEFAULT_FACE_MODEL = 'facenet_arcface_triplet_model6'" in app
    assert "facenet_proto: 'FaceNet Proto'" in app
    assert "facenet_contrastive_proto: 'FaceNet Contrastive Proto'" in app
    assert "facenet_contrastive_proto_model5: 'FaceNet Contrastive Proto Model 5'" in app
    assert "facenet_arcface_triplet_model6: 'FaceNet ArcFace Triplet Model 6'" in app
    assert "function applySelectedModelDefaultThreshold" in app
    assert "function formatThresholdInput" in app
    assert "facenet_proto: { threshold: 0.47 }" in app
    assert "facenet_contrastive_proto: { threshold: 0.800884 }" in app
    assert "facenet_contrastive_proto_model5: { threshold: 0.800884 }" in app
    assert "facenet_arcface_triplet_model6: { threshold: 0.3000000119 }" in app
    assert "Hybrid FaceNet, FaceNet Proto, FaceNet Contrastive Proto, FaceNet Contrastive Proto Model 5 ve FaceNet ArcFace Triplet Model 6" in i18n
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


def test_ui_navigation_and_state_wiring_regressions_are_pinned():
    index = (PROJECT_ROOT / "ui" / "index.html").read_text(encoding="utf-8")
    app = (PROJECT_ROOT / "ui" / "js" / "app.js").read_text(encoding="utf-8")

    assert 'data-nav-target="pending-review" data-review-filter="needs_review"' in index
    assert 'data-nav-target="flagged-attempts" data-review-filter="Rejected"' in index
    assert 'data-nav-target="audit-logs" data-admin-panel="operation-model-evidence"' in index
    assert 'data-nav-target="exam-settings" data-admin-panel="admin-policy-summary"' in index
    assert 'data-nav-target="verification-rules" data-admin-panel="admin-policy-summary"' in index

    assert 'id="notifications-btn"' in index
    assert 'id="help-btn"' in index
    assert 'id="simulation-view-all-btn"' in index
    for element_id in ["notifications-btn", "help-btn", "simulation-view-all-btn", "wrong-face-demo-btn"]:
        assert f"getElementById('{element_id}')" in app

    assert "Mark for Review" not in index
    assert "Request Fallback" in index
    assert 'id="view-launch"' not in index
    assert "launch:" not in app

    assert "function clearStudentInputState" in app
    assert app.count("clearStudentInputState({ enrollment: true, verification: true, simulation: true });") == 5
    assert "function isReviewActionAllowed" in app
    assert "REVIEW_ACTION_STATUSES = new Set(['Manual Review', 'Fallback Requested', 'Rejected'])" in app
    assert "SIMULATION_NO_ATTEMPT" in app
    assert "state.simulationScenario = scenario;" in app
    assert app.index("if (!targetStudentId)") < app.index("state.simulationScenario = scenario;")
    assert '<span class="top-context-pill"' in index
    assert '<button class="top-context-pill"' not in index

    select_start = app.index("async function selectAttempt")
    select_end = app.index("\nfunction renderReviewAttempt", select_start)
    select_body = app[select_start:select_end]
    assert "state.simulationSelectedAttemptId =" not in select_body
    assert "options.syncSimulationAttempt" in select_body
    assert "function attemptMatchesSimulationContext" in app
    assert "function syncSimulationAttemptSelection" in app
    assert "await selectAttempt(attemptId, { scroll: false, syncSimulationAttempt: true });" in app
    assert "await selectAttempt(attempt.attempt_id, { scroll: false, syncSimulationAttempt: true });" in app

    reset_start = app.index("async function resetDemo")
    reset_end = app.index("\nfunction resetDemoPresentationState", reset_start)
    reset_body = app[reset_start:reset_end]
    assert "await api.resetDemo();" in reset_body
    assert "resetDemoPresentationState();" in reset_body
    assert reset_body.index("resetDemoPresentationState();") < reset_body.index("await refreshAll();")
    assert "toast('Demo reset.', 'success');" in reset_body
    assert "flux_preupload" not in reset_body
    assert "imported_count" not in reset_body

    presentation_start = app.index("function resetDemoPresentationState")
    presentation_end = app.index("\nfunction resetStudentResultPanel", presentation_start)
    presentation_body = app[presentation_start:presentation_end]
    for marker in [
        "state.simulationScenario = 'matching';",
        "state.simulationSelectedAttemptId = null;",
        "state.simulationSelectedStudentId = null;",
        "setCheckedIfPresent('consent-check', false);",
        "setCheckedIfPresent('simulation-consent-check', false);",
        "setInputValueIfPresent('proctor-filter-status', '');",
        "setInputValueIfPresent('proctor-sort', 'priority');",
        "setTextIfPresent('flux-result', '');",
        "setStudentStep('consent');",
    ]:
        assert marker in presentation_body

    unwired_buttons = []
    for match in re.finditer(r"<button\b([^>]*)>", index, flags=re.IGNORECASE):
        attrs = match.group(1)
        if _button_attrs_are_wired(attrs):
            continue
        unwired_buttons.append(attrs.strip())
    assert _button_attrs_are_wired(' type="button" class="decorative"') is False
    assert unwired_buttons == []


def test_ui_sources_do_not_keep_mojibake_aliases():
    markers = ["Ä", "Ã", "Â", "\ufffd"]
    for relative_path in ["ui/js/i18n.js", "ui/index.html", "ui/js/app.js"]:
        text = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        found = [marker for marker in markers if marker in text]
        assert found == [], f"{relative_path} contains mojibake markers: {found}"
