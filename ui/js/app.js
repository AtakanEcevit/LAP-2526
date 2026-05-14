const state = {
    health: null,
    snapshot: null,
    roster: null,
    enrollments: [],
    selectedAttempt: null,
    selectedReviewStudent: null,
    selectedEnrollmentFiles: [],
    attemptHistory: [],
    auditRows: [],
    fluxStatus: null,
    fluxTestSet: null,
    stagedPreloadedSelfie: null,
    verificationInFlight: false,
    studentResultAttemptId: null,
    studentStep: 'consent',
    activeView: 'simulation',
    activeNavTarget: 'simulation',
    simulationScenario: 'matching',
    simulationMobilePane: 'student',
    simulationSelectedStudentId: null,
    simulationSelectedAttemptId: null,
    theme: 'dark'
};

const ENROLLMENT_TARGET = 3;
const ENROLLMENT_MAX = 5;
const DISPLAY_UNIVERSITY = 'Atılım Üniversitesi Demo';
const DISPLAY_COURSE = 'SE 204 - Veri Yapıları';
const DISPLAY_EXAM = 'Ara Sınav';
const DISPLAY_STUDENT_ID = 'AT-2026-1042';
const FALLBACK_DEMO_STUDENT_ID = 'NB-2026-1042';
const FALLBACK_DEMO_EXAM_ID = 'CS204-MIDTERM-1';
const DEFAULT_FACE_MODEL = 'facenet_arcface_triplet_model6';
const REVIEW_STATUSES = new Set(['Manual Review', 'Fallback Requested']);
const REVIEW_ACTION_STATUSES = new Set(['Manual Review', 'Fallback Requested', 'Rejected']);
const ACCESS_GRANTED_STATUSES = new Set(['Verified', 'Approved by Proctor']);
const BLOCKED_STATUSES = new Set(['Rejected']);
const WAITING_STATUSES = new Set(['Not Started', 'Enrollment Needed', 'Pending Verification']);
const STUDENT_STEP_ORDER = ['consent', 'enrollment', 'verification', 'result'];
const KPI_KEYS = ['attempts', 'verified', 'review', 'blocked', 'no-enrollment'];
const THEME_STORAGE_KEY = 'faceverifyTheme';
const SUPPORTED_THEMES = new Set(['light', 'dark']);
const SIMULATION_NO_ATTEMPT = '__none__';
const PERSONAS = {
    simulation: 'Canlı Simülasyon',
    student: 'Öğrenci: Aylin Kaya',
    proctor: 'İnceleme Masası: Proctor Lee',
    admin: 'Operasyonlar: Kayıt Yetkilisi',
    evidence: 'Operasyonlar: Model Kanıtı'
};

document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    bindTabs();
    bindActions();
    updateSidebarClock();
    window.setInterval(updateSidebarClock, 60000);
    showView('simulation');
    refreshAll();
    setStudentStep('consent');
});

function initTheme() {
    state.theme = sanitizeTheme(document.documentElement.dataset.theme || readStoredTheme());
    document.documentElement.dataset.theme = state.theme;
    document.querySelectorAll('[data-theme-option]').forEach(button => {
        button.addEventListener('click', () => setTheme(button.dataset.themeOption));
    });
    renderThemeToggle();
}

function readStoredTheme() {
    try {
        return window.localStorage.getItem(THEME_STORAGE_KEY);
    } catch (err) {
        return null;
    }
}

function sanitizeTheme(theme) {
    return SUPPORTED_THEMES.has(theme) ? theme : 'dark';
}

function setTheme(theme) {
    state.theme = sanitizeTheme(theme);
    document.documentElement.dataset.theme = state.theme;
    try {
        window.localStorage.setItem(THEME_STORAGE_KEY, state.theme);
    } catch (err) {
        // Theme switching should remain usable even when storage is blocked.
    }
    renderThemeToggle();
}

function renderThemeToggle() {
    document.querySelectorAll('[data-theme-option]').forEach(button => {
        const active = button.dataset.themeOption === state.theme;
        button.classList.toggle('active', active);
        button.setAttribute('aria-pressed', active ? 'true' : 'false');
    });
}

function bindTabs() {
    document.querySelectorAll('.tab-button').forEach(button => {
        button.addEventListener('click', () => activateNavigation(button));
    });
}

function activateNavigation(button) {
    const viewName = button.dataset.view;
    if (!viewName) return;
    if (button.dataset.reviewFilter !== undefined) {
        setReviewFilter(button.dataset.reviewFilter);
    }
    showView(viewName, {
        navTarget: button.dataset.navTarget || defaultNavTarget(viewName),
        adminPanel: button.dataset.adminPanel,
    });
}

function showView(viewName, options = {}) {
    const normalizedView = normalizeViewName(viewName);
    state.activeNavTarget = options.navTarget || defaultNavTarget(viewName, normalizedView);
    state.activeView = viewName === 'evidence' ? 'evidence' : normalizedView;
    document.querySelectorAll('.tab-button').forEach(button => {
        const target = button.dataset.navTarget || defaultNavTarget(button.dataset.view);
        const active = button.classList.contains('sidebar-link')
            ? target === state.activeNavTarget
            : button.dataset.view === normalizedView;
        button.classList.toggle('active', active);
    });
    document.querySelectorAll('.view').forEach(view => {
        view.classList.toggle('active', view.id === `view-${normalizedView}`);
    });
    updateScenarioRail(viewName);
    updateContextBar();
    if (normalizedView === 'simulation') {
        syncSimulationSelection();
        renderSimulation();
    }
    const adminPanel = options.adminPanel || ((viewName === 'evidence' || viewName === 'lab') ? 'operation-model-evidence' : null);
    if (adminPanel) {
        openOperationPanel(adminPanel);
    }
}

function bindActions() {
    document.querySelectorAll('[data-demo-jump]').forEach(button => {
        button.addEventListener('click', () => jumpFromGuide(button.dataset.demoJump));
    });
    document.getElementById('notifications-btn').addEventListener('click', () => {
        toast('Notifications panel is not available in this demo build yet.', 'info');
    });
    document.getElementById('help-btn').addEventListener('click', () => {
        toast('Use the scenario rail to move through policy, student gate, review, and audit views.', 'info');
    });
    document.getElementById('simulation-view-all-btn').addEventListener('click', () => {
        openReviewDesk('', 'proctor');
    });
    document.getElementById('student-refresh-btn').addEventListener('click', refreshAll);
    document.getElementById('proctor-refresh-btn').addEventListener('click', refreshAll);
    document.getElementById('simulation-refresh-btn').addEventListener('click', refreshAll);
    document.getElementById('consent-next-btn').addEventListener('click', () => {
        if (requireConsent()) setStudentStep('enrollment');
    });
    document.getElementById('enrollment-next-btn').addEventListener('click', () => {
        const context = enrollmentContext();
        if (context.modelMismatch) {
            toast(modelMismatchMessage(context), 'error');
            return;
        }
        if (context.sampleCount <= 0) {
            toast('Add at least one enrollment sample before verification.', 'error');
            return;
        }
        setStudentStep('verification');
    });
    document.getElementById('enroll-student-btn').addEventListener('click', enrollStudent);
    document.getElementById('verify-student-btn').addEventListener('click', verifyStudent);
    document.getElementById('simulation-verify-btn').addEventListener('click', verifySimulationStudent);
    document.getElementById('wrong-face-demo-btn').addEventListener('click', openWrongFaceReviewPath);
    document.getElementById('save-course-btn').addEventListener('click', saveCourse);
    document.getElementById('save-exam-btn').addEventListener('click', saveExam);
    document.getElementById('exam-model').addEventListener('change', applySelectedModelDefaultThreshold);
    document.getElementById('import-roster-btn').addEventListener('click', importRoster);
    document.getElementById('reset-demo-btn').addEventListener('click', resetDemo);
    document.getElementById('flux-preupload-btn').addEventListener('click', preuploadFlux);
    document.getElementById('flux-export-btn').addEventListener('click', exportFluxTestSet);
    document.getElementById('verify-preloaded-btn').addEventListener('click', () => stagePreloadedSelfie(false));
    document.getElementById('simulation-preloaded-btn').addEventListener('click', () => stagePreloadedSelfie(true));
    document.getElementById('lab-compare-btn').addEventListener('click', compareInLab);
    ['proctor-filter-status', 'proctor-filter-decision', 'proctor-filter-name', 'proctor-filter-id', 'proctor-sort']
        .forEach(id => document.getElementById(id).addEventListener('input', renderRoster));
    document.querySelectorAll('[data-review-action]').forEach(button => {
        button.addEventListener('click', () => reviewAttempt(button.dataset.reviewAction));
    });
    document.querySelectorAll('[data-simulation-scenario]').forEach(button => {
        button.addEventListener('click', () => setSimulationScenario(button.dataset.simulationScenario));
    });
    document.querySelectorAll('[data-simulation-review-action]').forEach(button => {
        button.addEventListener('click', () => reviewSimulationAttempt(button.dataset.simulationReviewAction));
    });
    document.querySelectorAll('[data-simulation-pane]').forEach(button => {
        button.addEventListener('click', () => {
            state.simulationMobilePane = button.dataset.simulationPane;
            renderSimulationPaneSwitch();
        });
    });
    document.getElementById('simulation-feed-grid').addEventListener('click', event => {
        const card = event.target.closest('[data-simulation-student-card]');
        if (!card) return;
        openSimulationFeedCard(card);
    });
    document.getElementById('simulation-feed-grid').addEventListener('keydown', event => {
        if (event.key !== 'Enter' && event.key !== ' ') return;
        const card = event.target.closest('[data-simulation-student-card]');
        if (!card) return;
        event.preventDefault();
        openSimulationFeedCard(card);
    });
    document.getElementById('student-exam').addEventListener('change', async () => {
        clearStudentInputState({ enrollment: true, verification: true, simulation: true });
        resetStudentResultPanel();
        await refreshRosterOnly();
        syncAdminFields();
        updateContextBar();
        renderEnrollmentGuidance();
        syncSimulationControls();
        renderSimulation();
    });
    document.getElementById('student-select').addEventListener('change', () => {
        clearStudentInputState({ enrollment: true, verification: true, simulation: true });
        resetStudentResultPanel();
        state.simulationSelectedAttemptId = null;
        updateContextBar();
        renderEnrollmentGuidance();
        syncSimulationSelection();
        renderSimulation();
    });
    document.getElementById('simulation-exam').addEventListener('change', async () => {
        clearStudentInputState({ enrollment: true, verification: true, simulation: true });
        setSelectValue('student-exam', valueOf('simulation-exam'));
        resetStudentResultPanel();
        await refreshRosterOnly();
        syncAdminFields();
        updateContextBar();
        renderEnrollmentGuidance();
        renderSimulation();
    });
    document.getElementById('simulation-student').addEventListener('change', () => {
        clearStudentInputState({ enrollment: true, verification: true, simulation: true });
        setSelectValue('student-select', valueOf('simulation-student'));
        resetStudentResultPanel();
        state.simulationSelectedAttemptId = null;
        syncSimulationSelection();
        updateContextBar();
        renderEnrollmentGuidance();
        renderSimulation();
    });
    document.getElementById('simulation-consent-check').addEventListener('change', renderSimulation);
    document.getElementById('simulation-verify-file').addEventListener('change', () => {
        clearStagedPreloadedSelfie('simulation');
        renderSimulation();
    });
    document.getElementById('enroll-files').addEventListener('change', renderEnrollmentFileState);
    document.getElementById('consent-check').addEventListener('change', updateStudentProgress);
    document.getElementById('verify-file').addEventListener('change', () => {
        clearStagedPreloadedSelfie('student');
        updateStudentProgress();
    });
    document.querySelectorAll('[data-student-step]').forEach(button => {
        button.addEventListener('click', () => setStudentStep(button.dataset.studentStep));
    });
}

async function refreshAll() {
    try {
        const [health, snapshot, audit, enrollments, fluxStatus, fluxTestSet] = await Promise.all([
            api.health(),
            api.snapshot(),
            api.audit(),
            api.users(),
            api.fluxStatus().catch(err => ({ available: false, error: err.message })),
            api.fluxTestSet().catch(err => ({ available: false, error: err.message }))
        ]);
        state.snapshot = snapshot;
        state.health = health;
        state.enrollments = enrollments;
        state.auditRows = audit;
        state.fluxStatus = fluxStatus;
        state.fluxTestSet = fluxTestSet;
        setStatus('All systems operational', true);
        populateSelectors();
        syncAdminFields();
        await refreshRosterOnly();
        renderEnrollmentGuidance();
        renderAudit(audit);
        renderFluxPanel(fluxStatus, fluxTestSet);
        syncSimulationSelection();
        renderSimulation();
        updateContextBar();
    } catch (err) {
        setStatus('Connection interrupted', false);
        renderSimulation();
        toast(err, 'error');
    }
}

async function refreshRosterOnly() {
    const examId = currentExamId();
    if (!examId) return;
    try {
        state.roster = await api.examRoster(examId);
        renderRoster();
        renderMetrics();
        renderEnrollmentGuidance();
        syncSimulationSelection();
        renderSimulation();
        updateContextBar();
    } catch (err) {
        toast(err, 'error');
    }
}

function clearStudentInputState({ enrollment = false, verification = false, simulation = false } = {}) {
    if (enrollment) {
        clearFileInput('enroll-files');
        state.selectedEnrollmentFiles = [];
        const previewGrid = document.getElementById('enroll-preview-grid');
        if (previewGrid) previewGrid.innerHTML = '';
    }
    if (verification) clearFileInput('verify-file');
    if (simulation) clearFileInput('simulation-verify-file');
    clearStagedPreloadedSelfie();
    if (enrollment && document.getElementById('enroll-files')) {
        renderEnrollmentFileState();
    } else {
        updateStudentProgress();
    }
}

function clearFileInput(id) {
    const input = document.getElementById(id);
    if (input) input.value = '';
}

function setInputValueIfPresent(id, value) {
    const input = document.getElementById(id);
    if (input) input.value = value;
}

function setCheckedIfPresent(id, checked) {
    const input = document.getElementById(id);
    if (input) input.checked = checked;
}

function openReviewDesk(filter = undefined, navTarget = 'proctor') {
    if (filter !== undefined) setReviewFilter(filter);
    showView('proctor', { navTarget });
}

function openWrongFaceReviewPath() {
    const row = currentRosterRow();
    const attempt = row?.latest_attempt;
    if (!isReviewActionAllowed(attempt)) {
        toast('Submit a wrong-face or low-confidence selfie first, then open the Review Desk.', 'info');
        return;
    }
    const finalStatus = attempt.final_status || attempt.status;
    const filter = BLOCKED_STATUSES.has(finalStatus) ? 'Rejected' : 'needs_review';
    const navTarget = BLOCKED_STATUSES.has(finalStatus) ? 'flagged-attempts' : 'pending-review';
    openReviewDesk(filter, navTarget);
    selectAttempt(attempt.attempt_id);
}

function jumpFromGuide(viewName) {
    setSelectValue('student-select', demoStudentId());
    if (viewName === 'student') {
        setStudentStep('enrollment');
        renderEnrollmentGuidance();
    }
    if (viewName === 'proctor') {
        openReviewDesk('needs_review', 'pending-review');
        return;
    }
    if (viewName === 'simulation') {
        syncSimulationSelection();
        renderSimulation();
    }
    showView(viewName, { navTarget: defaultNavTarget(viewName) });
}

function setStatus(text, ok) {
    const status = document.getElementById('system-status');
    document.getElementById('status-text').textContent = localizeText(text);
    status.classList.toggle('offline', !ok);
    status.classList.toggle('online', ok);
}

function updateSidebarClock() {
    const now = new Date();
    setTextIfPresent('sidebar-clock-time', now.toLocaleTimeString('tr-TR', {
        hour: '2-digit',
        minute: '2-digit'
    }));
    setTextIfPresent('sidebar-clock-date', now.toLocaleDateString('tr-TR', {
        month: 'short',
        day: 'numeric',
        year: 'numeric'
    }));
}

function populateSelectors() {
    if (!state.snapshot) return;
    const examSelect = document.getElementById('student-exam');
    const studentSelect = document.getElementById('student-select');
    const simulationExamSelect = document.getElementById('simulation-exam');
    const simulationStudentSelect = document.getElementById('simulation-student');
    const priorExam = examSelect.value;
    const priorStudent = studentSelect.value;
    const priorSimulationExam = simulationExamSelect.value;
    const priorSimulationStudent = simulationStudentSelect.value;

    const examOptions = state.snapshot.exams.map(exam =>
        `<option value="${escapeHtml(exam.exam_id)}">${escapeHtml(displayExamName(exam.name))} (${escapeHtml(exam.exam_id)})</option>`
    ).join('');
    const studentOptions = state.snapshot.students.map(student =>
        `<option value="${escapeHtml(student.student_id)}">${escapeHtml(student.name)} - ${escapeHtml(displayStudentId(student.student_id))}</option>`
    ).join('');

    examSelect.innerHTML = examOptions;
    studentSelect.innerHTML = studentOptions;
    simulationExamSelect.innerHTML = examOptions;
    simulationStudentSelect.innerHTML = studentOptions;

    setSelectValue('student-exam', priorExam || demoExamId());
    setSelectValue('student-select', priorStudent || demoStudentId());
    setSelectValue('simulation-exam', priorSimulationExam || examSelect.value);
    setSelectValue('simulation-student', priorSimulationStudent || studentSelect.value);
}

function syncAdminFields() {
    const course = state.snapshot.courses[0];
    const exam = selectedExam() || state.snapshot.exams[0];
    if (course) {
        document.getElementById('course-id').value = course.course_id;
        document.getElementById('course-name').value = displayCourseName(course.name);
        document.getElementById('course-instructor').value = displayInstructor(course.instructor);
        document.getElementById('course-term').value = displayTerm(course.term);
    }
    if (exam) {
        document.getElementById('exam-id').value = exam.exam_id;
        document.getElementById('exam-name').value = displayExamName(exam.name);
        document.getElementById('exam-start').value = toDatetimeLocal(exam.start_time);
        document.getElementById('exam-end').value = toDatetimeLocal(exam.end_time);
        document.getElementById('exam-threshold').value = exam.threshold;
        document.getElementById('exam-model').value = exam.model_type;
    }
}

function faceModelDefaults() {
    return {
        siamese: { threshold: 0.65 },
        prototypical: { threshold: 0.65 },
        hybrid: { threshold: 0.3000000119 },
        facenet_proto: { threshold: 0.47 },
        facenet_contrastive_proto: { threshold: 0.800884 },
        facenet_contrastive_proto_model5: { threshold: 0.800884 },
        facenet_arcface_triplet_model6: { threshold: 0.3000000119 },
        ...(state.health?.face_model_defaults || {})
    };
}

function modelDefaultThreshold(modelType) {
    const value = faceModelDefaults()[modelType]?.threshold;
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric : null;
}

function applySelectedModelDefaultThreshold() {
    const threshold = modelDefaultThreshold(valueOf('exam-model'));
    if (threshold === null) return;
    document.getElementById('exam-threshold').value = formatThresholdInput(threshold);
}

function formatThresholdInput(value) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return '';
    return String(Math.round(numeric * 1000000) / 1000000);
}

function renderMetrics() {
    const roster = state.roster?.roster || [];
    const metrics = dashboardMetrics(roster);

    setTextIfPresent('metric-students', roster.length);
    setTextIfPresent('metric-verified', metrics.verified);
    setTextIfPresent('metric-review', metrics.review);
    setTextIfPresent('sidebar-review-count', metrics.review);
    setTextIfPresent('sidebar-blocked-count', metrics.blocked);
    setTextIfPresent(
        'queue-summary',
        t('queue.summary', metrics)
    );
    renderKpiDeck('simulation', metrics);
    renderKpiDeck('review', metrics);
}

function dashboardMetrics(roster) {
    const rows = roster || [];
    return {
        roster: rows.length,
        attempts: rows.filter(row => row.latest_attempt).length,
        verified: rows.filter(row => ACCESS_GRANTED_STATUSES.has(row.exam_status)).length,
        review: rows.filter(row => REVIEW_STATUSES.has(row.exam_status)).length,
        waiting: rows.filter(row => WAITING_STATUSES.has(row.exam_status)).length,
        blocked: rows.filter(row => BLOCKED_STATUSES.has(row.exam_status)).length,
        noEnrollment: rows.filter(row =>
            row.exam_status === 'Enrollment Needed' || Number(row.sample_count || 0) <= 0
        ).length,
    };
}

function renderKpiDeck(prefix, metrics) {
    if (!document.getElementById(`${prefix}-kpi-attempts-count`)) return;
    const denominator = Math.max(1, metrics.roster || 0);
    const values = {
        attempts: metrics.attempts,
        verified: metrics.verified,
        review: metrics.review,
        blocked: metrics.blocked,
        'no-enrollment': metrics.noEnrollment,
    };

    KPI_KEYS.forEach(key => {
        const count = values[key] || 0;
        setTextIfPresent(`${prefix}-kpi-${key}-count`, count);
        setTextIfPresent(
            `${prefix}-kpi-${key}-rate`,
            key === 'attempts' ? localizeText('Live') : `${((count / denominator) * 100).toFixed(1)}%`
        );
        const spark = document.getElementById(`${prefix}-kpi-${key}-spark`);
        if (spark) {
            spark.innerHTML = sparklineSvg(kpiSparkValues(key, metrics), key);
        }
    });
}

function kpiSparkValues(key, metrics) {
    const total = Math.max(1, metrics.roster || 0);
    const end = key === 'attempts'
        ? metrics.attempts
        : key === 'verified'
            ? metrics.verified
            : key === 'review'
                ? metrics.review
                : key === 'blocked'
                    ? metrics.blocked
                    : metrics.noEnrollment;
    const values = [];
    for (let index = 0; index < 10; index += 1) {
        const progress = index / 9;
        const curve = Math.sin((index + key.length) * 1.3) * Math.max(1, total * 0.05);
        values.push(Math.max(0, Math.min(total, Math.round(end * progress + curve))));
    }
    return values;
}

function sparklineSvg(values, key) {
    const width = 140;
    const height = 34;
    const max = Math.max(1, ...values);
    const points = values.map((value, index) => {
        const x = (index / Math.max(1, values.length - 1)) * width;
        const y = height - (value / max) * (height - 4) - 2;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(' ');
    return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(t('sparkline.trend', { label: key }))}"><polyline points="${points}"></polyline></svg>`;
}

function renderRoster() {
    const body = document.getElementById('roster-table');
    const allRows = state.roster?.roster || [];
    const roster = filteredRosterRows(allRows);
    if (!allRows.length) {
        body.innerHTML = `<tr><td colspan="4" class="empty-cell">${escapeHtml(localizeText('No roster rows for this exam.'))}</td></tr>`;
        return;
    }
    if (!roster.length) {
        body.innerHTML = `<tr><td colspan="4" class="empty-cell">${escapeHtml(localizeText('No roster rows match these filters.'))}</td></tr>`;
        return;
    }
    body.innerHTML = roster.map(row => {
        const latest = row.latest_attempt;
        const score = latest ? latest.score.toFixed(3) : '-';
        const actionLabel = REVIEW_STATUSES.has(row.exam_status) ? localizeText('Review') : localizeText('Open');
        const action = latest
            ? `<button class="row-button icon-row-action ${REVIEW_STATUSES.has(row.exam_status) ? 'review-action' : ''}" data-attempt="${escapeHtml(latest.attempt_id)}" aria-label="${escapeHtml(actionLabel)}" title="${escapeHtml(actionLabel)}">${iconSvg('chevron')}</button>`
            : `<span class="muted action-waiting">${escapeHtml(localizeText('Waiting'))}</span>`;
        const selected = latest?.attempt_id === state.selectedAttempt?.attempt_id ? ' class="selected-row"' : '';
        return `
            <tr${selected}>
                <td>${studentNameCell(row)}</td>
                <td><span class="badge ${statusClass(row.exam_status)}">${escapeHtml(statusLabel(row.exam_status))}</span></td>
                <td>${score}</td>
                <td>${action}</td>
            </tr>
        `;
    }).join('');
    body.querySelectorAll('[data-attempt]').forEach(button => {
        button.addEventListener('click', () => selectAttempt(button.dataset.attempt));
    });
}

function filteredRosterRows(rows) {
    const status = valueOf('proctor-filter-status');
    const decision = valueOf('proctor-filter-decision');
    const name = valueOf('proctor-filter-name').toLowerCase();
    const studentId = valueOf('proctor-filter-id').toLowerCase();
    const sort = valueOf('proctor-sort') || 'priority';

    return rows
        .filter(row => {
            if (!status) return true;
            if (status === 'needs_review') return REVIEW_STATUSES.has(row.exam_status);
            return row.exam_status === status;
        })
        .filter(row => {
            const rowDecision = row.latest_attempt?.decision || 'none';
            return !decision || rowDecision === decision;
        })
        .filter(row => !name || String(row.name || '').toLowerCase().includes(name))
        .filter(row => !studentId || String(row.student_id || '').toLowerCase().includes(studentId))
        .sort((a, b) => sortRosterRows(a, b, sort));
}

function sortRosterRows(a, b, sort) {
    if (sort === 'name') return String(a.name || '').localeCompare(String(b.name || ''), 'tr-TR');
    if (sort === 'student_id') return String(a.student_id || '').localeCompare(String(b.student_id || ''), 'tr-TR');
    if (sort === 'score') {
        const aScore = Number.isFinite(Number(a.latest_attempt?.score)) ? Number(a.latest_attempt.score) : 2;
        const bScore = Number.isFinite(Number(b.latest_attempt?.score)) ? Number(b.latest_attempt.score) : 2;
        return aScore - bScore;
    }
    return statusPriority(a.exam_status) - statusPriority(b.exam_status)
        || String(a.name || '').localeCompare(String(b.name || ''), 'tr-TR');
}

function statusPriority(status) {
    if (REVIEW_STATUSES.has(status)) return 0;
    if (BLOCKED_STATUSES.has(status)) return 1;
    if (WAITING_STATUSES.has(status)) return 2;
    if (ACCESS_GRANTED_STATUSES.has(status)) return 3;
    return 4;
}

function setReviewFilter(status) {
    const select = document.getElementById('proctor-filter-status');
    select.value = status;
    renderRoster();
}

async function selectAttempt(attemptId, options = {}) {
    if (!attemptId) return;
    setReviewLoading();
    try {
        const result = await api.getAttempt(attemptId);
        const attempt = result.attempt;
        const history = await api.listAttempts(attempt.exam_id, attempt.student_id);
        const rosterStudent = (state.roster?.roster || []).find(row => row.student_id === attempt.student_id);
        const student = {
            ...(result.student || {}),
            ...(rosterStudent || {})
        };
        state.selectedAttempt = attempt;
        state.selectedReviewStudent = student;
        state.attemptHistory = history;
        if (options.syncSimulationAttempt) syncSimulationAttemptSelection(attempt);
        renderReviewAttempt(attempt, student, history);
        renderRoster();
        renderSimulation();
        if (options.scroll !== false) {
            const panel = document.getElementById('review-panel');
            panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
            panel.focus({ preventScroll: true });
        }
    } catch (err) {
        setReviewError(err.message);
        toast(err, 'error');
    }
}

function renderReviewAttempt(attempt, student, history = []) {
    document.getElementById('review-panel').classList.remove('review-empty');
    document.getElementById('review-title').textContent =
        `${student.name || attempt.student_name || attempt.student_id} - ${statusLabel(attempt.final_status || attempt.status)}`;
    document.getElementById('review-empty-state').textContent = localizeText('Attempt loaded. Review the images and status before taking action.');
    document.getElementById('review-status').textContent = statusLabel(attempt.final_status || attempt.status || '-');
    document.getElementById('review-decision').textContent = humanizeDecision(attempt.decision);
    document.getElementById('review-model').textContent = modelLabel(attempt.model_type);
    document.getElementById('review-score').textContent = numberOrDash(attempt.score);
    document.getElementById('review-threshold').textContent = numberOrDash(attempt.threshold);
    document.getElementById('review-attempt-source').textContent = attemptSourceLabel(attempt.attempt_source);
    document.getElementById('review-time').textContent = formatDateTime(attempt.timestamp);
    document.getElementById('review-warnings').textContent =
        attempt.warnings?.length ? attempt.warnings.join('; ') : localizeText('None');
    document.getElementById('review-previous-attempts').textContent =
        history.length > 1 ? localizeText(`${history.length - 1} earlier attempt(s) for this exam.`) : localizeText('No earlier attempts for this exam.');
    setPreview('reference-preview', student.reference_preview);
    setPreview('query-preview', attempt.query_preview);
    renderAuditRail('review-audit-rail', attempt, student);
    setReviewButtonsEnabled(attempt);
}

function setReviewLoading() {
    state.selectedAttempt = null;
    state.selectedReviewStudent = null;
    state.attemptHistory = [];
    document.getElementById('review-panel').classList.add('review-empty');
    document.getElementById('review-title').textContent = localizeText('Loading attempt...');
    document.getElementById('review-empty-state').textContent = localizeText('Loading attempt details and previews.');
    clearReviewDetails();
    setReviewButtonsEnabled(false);
}

function setReviewError(message) {
    state.selectedAttempt = null;
    state.selectedReviewStudent = null;
    state.attemptHistory = [];
    document.getElementById('review-panel').classList.add('review-empty');
    document.getElementById('review-title').textContent = localizeText('Attempt could not be loaded');
    document.getElementById('review-empty-state').textContent = message ? apiErrorMessage(message) : localizeText('Attempt could not be loaded.');
    clearReviewDetails();
    setReviewButtonsEnabled(false);
}

function resetReviewPanel() {
    state.selectedAttempt = null;
    state.selectedReviewStudent = null;
    state.attemptHistory = [];
    document.getElementById('review-panel').classList.add('review-empty');
    document.getElementById('review-title').textContent = localizeText('Select an attempt');
    document.getElementById('review-empty-state').textContent = localizeText('Select an attempt from the Needs Review queue or roster.');
    clearReviewDetails();
    setReviewButtonsEnabled(false);
}

function clearReviewDetails() {
    ['review-status', 'review-decision', 'review-model', 'review-score', 'review-threshold', 'review-attempt-source', 'review-time', 'review-warnings', 'review-previous-attempts']
        .forEach(id => {
            document.getElementById(id).textContent = '-';
        });
    setPreview('reference-preview', null);
    setPreview('query-preview', null);
    renderAuditRail('review-audit-rail', null, null);
}

function isReviewActionAllowed(attempt) {
    const status = attempt?.final_status || attempt?.status;
    return REVIEW_ACTION_STATUSES.has(status);
}

function setReviewButtonsEnabled(attempt) {
    const enabled = isReviewActionAllowed(attempt);
    document.querySelectorAll('[data-review-action]').forEach(button => {
        button.disabled = !enabled;
    });
}

function setPreview(id, dataUrl) {
    const host = document.getElementById(id);
    if (!host) return;
    host.innerHTML = dataUrl ? `<img src="${dataUrl}" alt="">` : localizeText('No image');
}

function setAvatar(id, dataUrl) {
    const host = document.getElementById(id);
    if (!host) return;
    host.innerHTML = dataUrl ? `<img src="${dataUrl}" alt="">` : '<span>FV</span>';
}

function studentNameCell(row) {
    return `
        <div class="student-face-cell">
            ${avatarHtml(row?.reference_preview)}
            <div>
                <strong>${escapeHtml(row?.name || '-')}</strong>
                <span class="row-subtext">${escapeHtml(displayStudentId(row?.student_id))}</span>
            </div>
        </div>
    `;
}

function avatarHtml(dataUrl) {
    return dataUrl
        ? `<span class="student-avatar"><img src="${dataUrl}" alt=""></span>`
        : '<span class="student-avatar empty">FV</span>';
}

async function enrollStudent() {
    const studentId = document.getElementById('student-select').value;
    const files = document.getElementById('enroll-files').files;
    const modelType = selectedExam()?.model_type || DEFAULT_FACE_MODEL;
    const context = enrollmentContext();
    if (!requireConsent()) return;
    if (!studentId || !files.length) {
        toast('Select a student and at least one enrollment image.', 'error');
        return;
    }
    if (context.modelMismatch) {
        toast(modelMismatchMessage(context), 'error');
        return;
    }
    if (files.length > context.remaining) {
        const message = localizeText(`You selected ${files.length} file(s), but only ${context.remaining} enrollment slot(s) remain.`);
        document.getElementById('enroll-warning').textContent = message;
        toast(message, 'error');
        return;
    }
    try {
        const result = await api.enrollStudent(studentId, modelType, files);
        toast(`Enrolled ${result.sample_count} sample(s).`, 'success');
        document.getElementById('enroll-files').value = '';
        state.selectedEnrollmentFiles = [];
        await refreshAll();
        renderEnrollmentGuidance();
        setStudentStep('verification');
    } catch (err) {
        document.getElementById('enroll-warning').textContent = apiErrorMessage(err);
        toast(err, 'error');
    }
}

function renderEnrollmentGuidance() {
    const context = enrollmentContext();
    const studentText = context.student
        ? `${context.student.name} - ${displayStudentId(context.student.student_id)}`
        : localizeText('Select a student');

    renderPolicySummaries();
    document.getElementById('enrollment-student-label').textContent = studentText;
    document.getElementById('enrollment-model-label').textContent =
        `${modelLabel(context.exam?.model_type || '-')} ${localizeText('enrollment for this exam')}`;
    document.getElementById('enrollment-count-label').textContent =
        `${Math.min(context.sampleCount, ENROLLMENT_TARGET)}/${ENROLLMENT_TARGET}`;
    document.getElementById('enrollment-remaining-label').textContent = context.remaining;

    const guidance = document.getElementById('enrollment-guidance');
    guidance.textContent = enrollmentGuidanceMessage(context);
    guidance.classList.toggle('warning', context.modelMismatch);
    guidance.classList.toggle('ready', !context.modelMismatch && context.sampleCount >= ENROLLMENT_TARGET);

    document.getElementById('enrollment-next-action').textContent = enrollmentNextAction(context);
    renderEnrollmentFileState();
    updateStudentProgress();
    updateContextBar();
}

function renderEnrollmentFileState() {
    const input = document.getElementById('enroll-files');
    const files = Array.from(input.files || []);
    state.selectedEnrollmentFiles = files;

    const context = enrollmentContext();
    const summary = document.getElementById('enroll-selected-summary');
    const warning = document.getElementById('enroll-warning');
    const button = document.getElementById('enroll-student-btn');
    const previewGrid = document.getElementById('enroll-preview-grid');

    summary.textContent = files.length
        ? localizeText(`${files.length} file(s) selected. ${context.remaining} enrollment slot(s) left.`)
        : localizeText('No files selected.');

    if (context.modelMismatch) {
        warning.textContent = modelMismatchMessage(context);
    } else if (files.length > context.remaining) {
        warning.textContent = localizeText(`You selected ${files.length} file(s), but only ${context.remaining} enrollment slot(s) remain.`);
    } else {
        warning.textContent = '';
    }

    button.textContent = files.length
        ? localizeText(`Add ${files.length} Sample${files.length === 1 ? '' : 's'}`)
        : localizeText('Add Face Samples');
    button.disabled = !files.length || files.length > context.remaining || context.modelMismatch;

    previewGrid.innerHTML = '';
    files.slice(0, ENROLLMENT_MAX).forEach((file, index) => {
        const url = URL.createObjectURL(file);
        const item = document.createElement('div');
        item.className = 'selected-preview';
        item.innerHTML = `<img src="${url}" alt=""><span>${index + 1}</span>`;
        const img = item.querySelector('img');
        img.addEventListener('load', () => URL.revokeObjectURL(url), { once: true });
        previewGrid.appendChild(item);
    });

    updateStudentProgress();
}

function enrollmentContext() {
    const student = selectedStudent();
    const exam = selectedExam();
    const rosterRow = currentRosterRow();
    const enrollment = selectedCampusEnrollment();
    const sampleCount = Number(enrollment?.sample_count ?? rosterRow?.sample_count ?? student?.sample_count ?? 0);
    const modelMismatch = Boolean(
        enrollment && exam?.model_type && enrollment.model_type !== exam.model_type
    );

    return {
        student,
        exam,
        rosterRow,
        enrollment,
        sampleCount,
        remaining: Math.max(0, ENROLLMENT_MAX - sampleCount),
        modelMismatch
    };
}

function enrollmentGuidanceMessage(context) {
    if (!context.student) return localizeText('Select a student to view enrollment guidance.');
    if (context.modelMismatch) return modelMismatchMessage(context);
    if (context.sampleCount <= 0) return localizeText('Add 3 clear face samples before verification.');
    if (context.sampleCount < ENROLLMENT_TARGET) {
        const remainingRecommended = ENROLLMENT_TARGET - context.sampleCount;
        return localizeText(`Enrollment works, but add ${remainingRecommended} more sample${remainingRecommended === 1 ? '' : 's'} for a stronger prototype.`);
    }
    return localizeText('Enrollment ready. Continue to exam-day selfie.');
}

function enrollmentNextAction(context) {
    if (context.modelMismatch) return localizeText('This model needs a separate enrollment before verification.');
    if (context.sampleCount <= 0) return localizeText('Add 3 clear samples before submitting the exam-day selfie.');
    if (context.sampleCount < ENROLLMENT_TARGET) {
        return localizeText(`Add ${ENROLLMENT_TARGET - context.sampleCount} more sample(s), or continue with lower reliability for demo purposes.`);
    }
    return localizeText('Enrollment is ready. Submit the exam-day selfie to request exam access.');
}

function modelMismatchMessage(context) {
    return localizeText(`This exam uses ${modelLabel(context.exam?.model_type)}; re-enroll for this model.`);
}

async function verifyStudent() {
    const examId = currentExamId();
    const studentId = document.getElementById('student-select').value;
    const file = document.getElementById('verify-file').files[0];
    if (!requireConsent()) return;
    if (!examId || !studentId) {
        toast('Select an exam and student first.', 'error');
        return;
    }
    if (!file && !isCurrentStagedPreloaded('student', examId, studentId)) {
        toast('Select an exam, student, and exam-day selfie.', 'error');
        return;
    }
    if (!confirmVerification(examId, studentId, file ? 'upload' : 'preloaded_demo', file?.name)) return;
    try {
        setVerificationInFlight(true);
        const result = file
            ? await api.verifyStudent(examId, studentId, file)
            : await api.verifyPreloadedStudent(examId, studentId, state.stagedPreloadedSelfie.scenario, true);
        renderStudentResult(result);
        document.getElementById('verify-file').value = '';
        clearStagedPreloadedSelfie();
        await refreshAll();
    } catch (err) {
        if (err.message.includes('enrolled with') || err.message.includes('requires')) {
            document.getElementById('enroll-warning').textContent = apiErrorMessage(err);
            renderEnrollmentGuidance();
        }
        toast(err, 'error');
    } finally {
        setVerificationInFlight(false);
    }
}

function stagePreloadedSelfie(fromSimulation = false) {
    const examId = currentExamId();
    const studentId = fromSimulation
        ? document.getElementById('simulation-student').value
        : document.getElementById('student-select').value;
    if (!examId || !studentId) {
        toast('Select an exam and student first.', 'error');
        return;
    }
    const row = (state.roster?.roster || []).find(item => item.student_id === studentId)
        || state.snapshot?.students?.find(student => student.student_id === studentId);
    if (row?.face_source !== 'flux_synid') {
        toast('This student does not have a preloaded FLUXSynID selfie.', 'error');
        return;
    }
    if (fromSimulation) {
        setSelectValue('student-select', studentId);
    }
    state.stagedPreloadedSelfie = {
        examId,
        studentId,
        scenario: 'matching',
        fromSimulation,
        imageUrl: api.studentFluxTestImageUrl(studentId),
    };
    document.getElementById(fromSimulation ? 'simulation-verify-file' : 'verify-file').value = '';
    renderStagedPreloadedSelfie();
    toast('Preloaded demo selfie staged. Confirm verification to submit.', 'success');
    renderSimulation();
}

function isCurrentStagedPreloaded(surface, examId, studentId) {
    const staged = state.stagedPreloadedSelfie;
    if (!staged) return false;
    const surfaceMatches = surface === 'simulation'
        ? staged.fromSimulation
        : !staged.fromSimulation;
    return surfaceMatches && staged.examId === examId && staged.studentId === studentId;
}

function clearStagedPreloadedSelfie(surface = null) {
    if (!surface || isCurrentStagedPreloaded(surface, state.stagedPreloadedSelfie?.examId, state.stagedPreloadedSelfie?.studentId)) {
        state.stagedPreloadedSelfie = null;
    }
    renderStagedPreloadedSelfie();
}

function renderStagedPreloadedSelfie() {
    const staged = state.stagedPreloadedSelfie;
    const student = staged
        ? state.snapshot?.students?.find(item => item.student_id === staged.studentId)
        : null;
    const text = staged
        ? localizeText(`Preloaded synthetic selfie staged for ${student?.name || displayStudentId(staged.studentId)}. Press Verify for Exam Access to submit.`)
        : '';

    const studentNote = document.getElementById('preloaded-stage-note');
    if (studentNote) {
        studentNote.textContent = !staged?.fromSimulation ? text : '';
        studentNote.classList.toggle('hidden', !staged || staged.fromSimulation);
    }

    const simulationNote = document.getElementById('simulation-preloaded-stage-note');
    if (simulationNote) {
        simulationNote.textContent = staged?.fromSimulation ? text : '';
        simulationNote.classList.toggle('hidden', !staged || !staged.fromSimulation);
    }

    if (staged?.fromSimulation) {
        setPreview('simulation-student-camera-preview', staged.imageUrl);
    }
}

function confirmVerification(examId, studentId, source, filename = '') {
    const student = state.snapshot?.students?.find(item => item.student_id === studentId);
    const exam = state.snapshot?.exams?.find(item => item.exam_id === examId);
    const sourceText = source === 'preloaded_demo'
        ? t('source.preloaded')
        : (filename ? t('source.uploadedNamed', { filename }) : t('source.uploaded'));
    const confirmed = window.confirm(
        t('confirm.submitVerification', {
            student: student?.name || studentId,
            exam: displayExamName(exam?.name) || examId,
            source: sourceText,
        })
    );
    if (!confirmed) {
        toast('Verification cancelled. No attempt was recorded.', 'info');
    }
    return confirmed;
}

function setVerificationInFlight(inFlight) {
    state.verificationInFlight = inFlight;
    [
        'verify-student-btn',
        'simulation-verify-btn',
        'verify-preloaded-btn',
        'simulation-preloaded-btn',
    ].forEach(id => {
        const button = document.getElementById(id);
        if (button) button.disabled = inFlight;
    });
    const simulationView = document.getElementById('view-simulation');
    if (simulationView) {
        simulationView.setAttribute('aria-busy', inFlight ? 'true' : 'false');
    }
    document.querySelectorAll('.student-pov, .simulation-result').forEach(el => {
        el.classList.toggle('is-loading', inFlight);
    });
}

function renderStudentResult(result) {
    const attempt = result.attempt;
    state.studentResultAttemptId = attempt.attempt_id;
    const panel = document.getElementById('student-result');
    panel.classList.remove('verified', 'review', 'rejected');
    panel.classList.add(resultClass(attempt.decision));
    document.getElementById('result-title').textContent = studentFacingTitle(attempt);
    document.getElementById('result-message').textContent = attempt.decision === 'verified'
        ? localizeText('Model verified. Access can proceed if the exam policy allows it.')
        : localizeText('Use proctor review or manual ID fallback before granting access.');
    document.getElementById('result-score').textContent = attempt.score.toFixed(3);
    document.getElementById('result-threshold').textContent = attempt.threshold.toFixed(3);
    document.getElementById('result-attempt-id').textContent = attempt.attempt_id;
    document.getElementById('result-time').textContent = formatDateTime(attempt.timestamp);
    updateScoreBar('result', attempt.score, attempt.threshold, attempt.decision);
    setStudentStep('result');
    updateStudentProgress();
}

async function saveCourse() {
    const course = {
        course_id: valueOf('course-id'),
        name: valueOf('course-name'),
        instructor: valueOf('course-instructor'),
        term: valueOf('course-term')
    };
    try {
        await api.saveCourse(course);
        toast('Course saved.', 'success');
        await refreshAll();
    } catch (err) {
        toast(err, 'error');
    }
}

async function saveExam() {
    const exam = {
        exam_id: valueOf('exam-id'),
        course_id: valueOf('course-id'),
        name: valueOf('exam-name'),
        start_time: valueOf('exam-start'),
        end_time: valueOf('exam-end'),
        threshold: valueOf('exam-threshold'),
        model_type: valueOf('exam-model')
    };
    const previous = state.snapshot?.exams?.find(item => item.exam_id === exam.exam_id);
    try {
        await api.saveExam(exam);
        const modelChanged = previous && previous.model_type !== exam.model_type;
        toast(
            modelChanged
                ? `Exam now uses ${modelLabel(exam.model_type)}. Students enrolled under ${modelLabel(previous.model_type)} must re-enroll.`
                : 'Exam saved.',
            'success'
        );
        await refreshAll();
    } catch (err) {
        toast(err, 'error');
    }
}

async function importRoster() {
    const file = document.getElementById('roster-file').files[0];
    const courseId = valueOf('course-id');
    if (!file) {
        toast('Choose a CSV roster file.', 'error');
        return;
    }
    try {
        const result = await api.importRoster(courseId, file);
        document.getElementById('import-result').textContent =
            localizeText(`Imported ${result.imported.length}; rejected ${result.rejected.length}.`);
        toast('Roster import complete.', 'success');
        await refreshAll();
    } catch (err) {
        toast(err, 'error');
    }
}

function renderFluxPanel(status = state.fluxStatus, testSet = state.fluxTestSet) {
    if (!document.getElementById('flux-status')) return;
    const available = Boolean(status?.available);
    const preuploaded = Number(status?.preuploaded_students || 0);
    const eligible = Number(status?.eligible_identity_count || 0);
    const configuredPath = status?.configured_path || '';
    const exported = Number(testSet?.image_count || 0);

    if (!valueOf('flux-dataset-dir') && configuredPath) {
        document.getElementById('flux-dataset-dir').value = configuredPath;
    }

    document.getElementById('flux-status').textContent = available
        ? localizeText(`${eligible} eligible identities found | ${preuploaded} students preuploaded`)
        : apiErrorMessage(status?.error || 'FLUXSynID dataset not found on this machine.');
    document.getElementById('flux-status').classList.toggle('warning', !available);
    document.getElementById('flux-status').classList.toggle('ready', available && preuploaded > 0);
    document.getElementById('flux-normalized-path').textContent = status?.normalized_path || '-';
    document.getElementById('flux-test-output').textContent = testSet?.output_dir || '-';
    document.getElementById('flux-test-count').textContent =
        localizeText(`${exported} matching selfie${exported === 1 ? '' : 'ies'} exported`);

    const downloadLink = document.getElementById('flux-download-zip');
    downloadLink.href = api.fluxTestSetZipUrl();
    downloadLink.classList.toggle('hidden', exported <= 0);
}

async function preuploadFlux() {
    const button = document.getElementById('flux-preupload-btn');
    const priorText = button.textContent;
    button.disabled = true;
    button.textContent = localizeText('Preuploading...');
    try {
        const result = await api.preuploadFlux({
            dataset_dir: valueOf('flux-dataset-dir'),
            count: valueOf('flux-count') || 25,
            seed: valueOf('flux-seed') || 42,
            model_type: valueOf('flux-model') || DEFAULT_FACE_MODEL
        });
        state.fluxStatus = result.status;
        state.fluxTestSet = result.export;
        renderFluxPanel(result.status, result.export);
        document.getElementById('flux-result').textContent =
            localizeText(`Preuploaded ${result.imported_count} student(s); exported ${result.export?.image_count || 0} test selfie file(s); skipped ${result.skipped.length}.`);
        toast('FLUXSynID preupload complete.', 'success');
        await refreshAll();
    } catch (err) {
        document.getElementById('flux-result').textContent = apiErrorMessage(err);
        toast(err, 'error');
    } finally {
        button.disabled = false;
        button.textContent = priorText;
    }
}

async function exportFluxTestSet() {
    const button = document.getElementById('flux-export-btn');
    const priorText = button.textContent;
    button.disabled = true;
    button.textContent = localizeText('Exporting...');
    try {
        const result = await api.exportFluxTestSet();
        state.fluxTestSet = result;
        renderFluxPanel(state.fluxStatus, result);
        document.getElementById('flux-result').textContent =
            localizeText(`Exported ${result.image_count} matching selfie file(s); skipped ${result.skipped.length}.`);
        toast('FLUXSynID test selfies exported.', 'success');
        renderGateContextPanel();
    } catch (err) {
        document.getElementById('flux-result').textContent = apiErrorMessage(err);
        toast(err, 'error');
    } finally {
        button.disabled = false;
        button.textContent = priorText;
    }
}

async function resetDemo() {
    if (!window.confirm(t('confirm.resetDemo'))) return;
    try {
        await api.resetDemo();
        resetDemoPresentationState();
        toast('Demo reset.', 'success');
        await refreshAll();
    } catch (err) {
        toast(err, 'error');
    }
}

function resetDemoPresentationState() {
    clearStudentInputState({ enrollment: true, verification: true, simulation: true });
    resetReviewPanel();
    resetStudentResultPanel();
    state.selectedAttempt = null;
    state.selectedReviewStudent = null;
    state.attemptHistory = [];
    state.simulationScenario = 'matching';
    state.simulationSelectedAttemptId = null;
    state.simulationSelectedStudentId = null;
    setCheckedIfPresent('consent-check', false);
    setCheckedIfPresent('simulation-consent-check', false);
    setInputValueIfPresent('proctor-filter-status', '');
    setInputValueIfPresent('proctor-filter-decision', '');
    setInputValueIfPresent('proctor-filter-name', '');
    setInputValueIfPresent('proctor-filter-id', '');
    setInputValueIfPresent('proctor-sort', 'priority');
    setTextIfPresent('flux-result', '');
    setStudentStep('consent');
    renderSimulation();
}

function resetStudentResultPanel() {
    state.studentResultAttemptId = null;
    const panel = document.getElementById('student-result');
    panel.classList.remove('verified', 'review', 'rejected', 'active');
    document.getElementById('result-title').textContent = localizeText('No attempt yet');
    document.getElementById('result-message').textContent = localizeText('Select a student, consent, and submit a selfie.');
    document.getElementById('result-score').textContent = '-';
    document.getElementById('result-threshold').textContent = '-';
    document.getElementById('result-attempt-id').textContent = '-';
    document.getElementById('result-time').textContent = '-';
    updateScoreBar('result', 0, selectedExam()?.threshold || 0, 'rejected');
    if (state.studentStep === 'result') setStudentStep('verification');
    updateStudentProgress();
}

async function reviewAttempt(action) {
    if (!state.selectedAttempt) {
        toast('Open an attempt before reviewing.', 'error');
        return;
    }
    if (!isReviewActionAllowed(state.selectedAttempt)) {
        toast('This attempt is not eligible for manual review.', 'error');
        return;
    }
    const attemptId = state.selectedAttempt.attempt_id;
    try {
        await api.reviewAttempt(
            attemptId,
            valueOf('reviewer-name'),
            action,
            valueOf('review-reason')
        );
        toast('Review saved.', 'success');
        await refreshAll();
        await selectAttempt(attemptId, { scroll: false });
    } catch (err) {
        toast(err, 'error');
    }
}

function renderAudit(rows) {
    const body = document.getElementById('audit-table');
    const latest = [...rows].reverse().slice(0, 12);
    body.innerHTML = latest.map(row => `
        <tr>
            <td>${escapeHtml(formatDateTime(row.timestamp))}</td>
            <td>${escapeHtml(eventLabel(row.event_type))}</td>
            <td>${escapeHtml(row.actor)}</td>
            <td>${escapeHtml(localizeText(row.message))}</td>
        </tr>
    `).join('');
}

function syncSimulationControls() {
    setSelectValue('simulation-exam', currentExamId());
    setSelectValue('simulation-student', document.getElementById('student-select').value);
}

function syncSimulationSelection() {
    const selectedStudentId = document.getElementById('student-select').value || demoStudentId();
    state.simulationSelectedStudentId = selectedStudentId;
    syncSimulationControls();
}

function setSimulationScenario(scenario) {
    const targetStudentId = findSimulationScenarioStudent(scenario);
    if (!targetStudentId) {
        renderSimulation();
        return;
    }
    state.simulationScenario = scenario;
    if (targetStudentId) {
        setSelectValue('student-select', targetStudentId);
        setSelectValue('simulation-student', targetStudentId);
        state.simulationSelectedStudentId = targetStudentId;
        resetStudentResultPanel();
        updateContextBar();
        renderEnrollmentGuidance();
    }

    const targetAttempt = currentRosterRow()?.latest_attempt;
    if (targetAttempt) {
        state.simulationSelectedAttemptId = targetAttempt.attempt_id;
    } else if (scenario === 'matching' || scenario === 'no_enrollment' || scenario === 'model_mismatch') {
        state.simulationSelectedAttemptId = SIMULATION_NO_ATTEMPT;
    }

    renderSimulation();
}

function findSimulationScenarioStudent(scenario) {
    const rows = state.roster?.roster || [];
    const exam = selectedExam();
    if (!rows.length) return null;

    if (scenario === 'matching') {
        return rows.find(row => row.student_id === demoStudentId())?.student_id || rows[0]?.student_id;
    }

    if (scenario === 'needs_review') {
        const row = rows.find(item => REVIEW_STATUSES.has(item.exam_status));
        if (!row) toast('No manual-review attempt exists yet. Submit a low-confidence or wrong-face selfie to create one.', 'info');
        return row?.student_id || null;
    }

    if (scenario === 'no_enrollment') {
        const row = rows.find(item => item.exam_status === 'Enrollment Needed' || Number(item.sample_count || 0) <= 0);
        if (!row) toast('Every rostered student is currently enrolled. Reset the demo or import a new roster row to show this path.', 'info');
        return row?.student_id || null;
    }

    if (scenario === 'model_mismatch') {
        const enrollment = state.enrollments.find(item =>
            item.user_id?.startsWith('campus_') && exam?.model_type && item.model_type !== exam.model_type
        );
        const studentId = enrollment?.user_id?.replace(/^campus_/, '');
        const row = rows.find(item => item.student_id === studentId);
        if (!row) toast('No model-mismatch enrollment exists for this exam yet. Change the exam model after enrolling a student to show this path.', 'info');
        return row?.student_id || null;
    }

    if (scenario === 'fallback_requested') {
        const row = rows.find(item => item.exam_status === 'Fallback Requested');
        if (!row) toast('No fallback request exists yet. Use Request Fallback after opening a review attempt.', 'info');
        return row?.student_id || null;
    }

    return null;
}

function renderSimulation() {
    if (!document.getElementById('view-simulation')) return;
    syncSimulationControls();
    renderSimulationStudentPane();
    renderSimulationInstructorPane();
    renderSimulationFeed();
    renderSimulationTimeline();
    document.querySelectorAll('[data-simulation-scenario]').forEach(button => {
        button.classList.toggle('active', button.dataset.simulationScenario === state.simulationScenario);
    });
    renderSimulationPaneSwitch();
}

function renderSimulationStudentPane() {
    const context = enrollmentContext();
    const student = context.student;
    const exam = context.exam;
    const row = context.rosterRow;
    const attempt = simulationAttempt();
    const status = row?.exam_status || (context.sampleCount > 0 ? 'Pending Verification' : 'Enrollment Needed');
    const consentDone = document.getElementById('simulation-consent-check').checked;
    const enrollmentDone = !context.modelMismatch && context.sampleCount >= ENROLLMENT_TARGET;
    const verificationDone = Boolean(attempt);
    const staged = isCurrentStagedPreloaded('simulation', currentExamId(), student?.student_id);

    setTextIfPresent('simulation-student-title', `${displayExamName(exam?.name)} access gate`);
    setTextIfPresent('simulation-student-id', displayStudentId(student?.student_id || '-'));
    setTextIfPresent('simulation-student-name', student?.name || 'Select a student');
    setAvatar('simulation-student-avatar', row?.reference_preview || student?.reference_preview);
    setTextIfPresent('simulation-exam-window', formatExamWindow(exam));
    setSimulationBadge('simulation-student-status', status);
    setSimulationMiniStep('simulation-mini-consent', consentDone, !consentDone);
    setSimulationMiniStep('simulation-mini-enrollment', enrollmentDone, consentDone && !enrollmentDone);
    setSimulationMiniStep('simulation-mini-verification', verificationDone, enrollmentDone && !verificationDone);
    setSimulationMiniStep('simulation-mini-result', verificationDone, verificationDone);
    renderGateHeader(
        'simulation',
        `${displayExamName(exam?.name)} access gate`,
        gateStatusInfo(context, attempt || row?.latest_attempt || null),
        simulationGateStep(consentDone, enrollmentDone, verificationDone)
    );
    setTextIfPresent(
        'simulation-student-guidance',
        staged
            ? localizeText('Preloaded demo selfie is staged. Press Verify for Exam Access to submit after confirmation.')
            : simulationStudentGuidance(context, attempt)
    );
    setPreview(
        'simulation-student-camera-preview',
        attempt?.query_preview || (staged ? state.stagedPreloadedSelfie.imageUrl : null)
    );
    renderStagedPreloadedSelfie();

    const resultCard = document.getElementById('simulation-result-card');
    resultCard.classList.remove('verified', 'review', 'rejected');
    if (attempt) {
        resultCard.classList.add(resultClass(attempt.decision));
        setTextIfPresent('simulation-result-title', studentFacingTitle(attempt));
        setTextIfPresent('simulation-result-message', studentFacingMessage(attempt));
        setTextIfPresent('simulation-result-score', numberOrDash(attempt.score));
        setTextIfPresent('simulation-result-threshold', numberOrDash(attempt.threshold));
        setTextIfPresent('simulation-result-attempt', attempt.attempt_id);
        updateScoreBar('simulation', attempt.score, attempt.threshold, attempt.decision);
    } else {
        setTextIfPresent('simulation-result-title', 'No attempt yet');
        setTextIfPresent('simulation-result-message', 'The student has not submitted an exam-day selfie in this simulation.');
        setTextIfPresent('simulation-result-score', '-');
        setTextIfPresent('simulation-result-threshold', numberOrDash(exam?.threshold));
        setTextIfPresent('simulation-result-attempt', '-');
        updateScoreBar('simulation', 0, exam?.threshold || 0, 'rejected');
    }
}

function renderSimulationInstructorPane() {
    const exam = selectedExam();
    const course = selectedCourse();
    const student = selectedStudent();
    const row = currentRosterRow();
    const attempt = simulationAttempt();
    const roster = state.roster?.roster || [];
    const metrics = dashboardMetrics(roster);
    const threshold = Number(exam?.threshold);
    const thresholdText = Number.isFinite(threshold) ? threshold.toFixed(3) : '-';

    setTextIfPresent('simulation-admin-title', `${courseCode(displayCourseName(course?.name))} operations cockpit`);
    setSimulationBadge('simulation-admin-model', modelLabel(exam?.model_type || '-'), 'model');
    renderKpiDeck('simulation', metrics);
    setTextIfPresent('simulation-policy-title', `${modelLabel(exam?.model_type || '-')} verification before exam access`);
    setTextIfPresent('simulation-policy-detail', `Threshold ${thresholdText}. Window ${formatExamWindow(exam)}. Instructor ${displayInstructor(course?.instructor)}.`);
    setTextIfPresent('simulation-admin-student', student ? `${student.name} - ${displayStudentId(student.student_id)}` : '-');
    setSimulationBadge('simulation-admin-status', row?.exam_status || '-');
    setTextIfPresent('simulation-admin-decision', humanizeDecision(attempt?.decision));
    setTextIfPresent('simulation-admin-warnings', attempt?.warnings?.length ? attempt.warnings.join('; ') : 'None');
    setTextIfPresent('simulation-admin-time', formatDateTime(attempt?.timestamp));
    setPreview('simulation-reference-preview', row?.reference_preview || student?.reference_preview);
    setPreview('simulation-query-preview', attempt?.query_preview);
    renderAuditRail('simulation-audit-rail', attempt, row);
    renderSimulationEvidence(attempt, row, exam);

    document.querySelectorAll('[data-simulation-review-action]').forEach(button => {
        button.disabled = !isReviewActionAllowed(attempt);
    });
}

function renderSimulationEvidence(attempt, row, exam) {
    const score = Number(attempt?.score);
    const threshold = Number(attempt?.threshold ?? exam?.threshold);
    const scorePercent = Number.isFinite(score) ? `${(score * 100).toFixed(1)}%` : '-';
    const thresholdText = Number.isFinite(threshold) ? threshold.toFixed(3) : '-';
    const source = attempt ? attemptSourceLabel(attempt.attempt_source) : 'No submission';
    const model = modelLabel(attempt?.model_type || exam?.model_type || '-');
    const samples = Number(row?.sample_count || 0);
    const scenario = attempt?.scenario ? humanizeScenario(attempt.scenario) : humanizeScenario(state.simulationScenario);
    const attemptId = attempt?.attempt_id || '-';

    setTextIfPresent('simulation-evidence-score', scorePercent);
    setTextIfPresent('simulation-evidence-score-detail', attempt ? scoreDetail(attempt) : 'Awaiting attempt');
    setTextIfPresent('simulation-evidence-threshold', thresholdText);
    setTextIfPresent('simulation-evidence-source', source);
    setTextIfPresent('simulation-evidence-model', model);
    setTextIfPresent('simulation-evidence-samples', samples ? `${samples} enrolled` : '-');
    setTextIfPresent('simulation-evidence-scenario', scenario);
    setTextIfPresent('simulation-evidence-attempt', attemptId === '-' ? '-' : shortAttemptId(attemptId));

    const fill = document.getElementById('simulation-evidence-score-fill');
    const marker = document.getElementById('simulation-evidence-threshold-marker');
    if (fill) {
        fill.style.width = `${Math.max(0, Math.min(1, Number.isFinite(score) ? score : 0)) * 100}%`;
    }
    if (marker) {
        marker.style.left = `${Math.max(0, Math.min(1, Number.isFinite(threshold) ? threshold : 0)) * 100}%`;
    }

    const rail = document.getElementById('simulation-evidence-rail');
    if (!rail) return;
    const finalStatus = attempt?.final_status || row?.exam_status || 'Pending Verification';
    const warnings = attempt?.warnings || [];
    const stages = attempt
        ? [
            { label: 'Source', value: source, kind: 'passed' },
            { label: 'Liveness', value: warnings.length ? 'Warnings' : 'Passed', kind: warnings.length ? 'review' : 'passed' },
            { label: 'Match', value: scorePercent, kind: resultClass(attempt.decision) },
            { label: 'Outcome', value: statusLabel(finalStatus), kind: statusClass(finalStatus) },
        ]
        : [
            { label: 'Source', value: 'Waiting', kind: 'waiting' },
            { label: 'Liveness', value: 'Pending', kind: 'waiting' },
            { label: 'Match', value: '-', kind: 'waiting' },
            { label: 'Outcome', value: statusLabel(finalStatus), kind: statusClass(finalStatus) },
        ];
    rail.innerHTML = stages.map(stage => `
        <div class="simulation-evidence-step ${escapeHtml(stage.kind)}">
            <strong>${escapeHtml(stage.label)}</strong>
            <small>${escapeHtml(stage.value)}</small>
        </div>
    `).join('');
}

function shortAttemptId(attemptId) {
    const value = String(attemptId || '');
    if (value.length <= 12) return value;
    return `${value.slice(0, 8)}...${value.slice(-4)}`;
}

function humanizeScenario(scenario) {
    if (!scenario) return '-';
    return String(scenario)
        .replace(/_/g, ' ')
        .replace(/\b\w/g, char => char.toUpperCase());
}

function renderSimulationFeed() {
    const grid = document.getElementById('simulation-feed-grid');
    if (!grid) return;
    const exam = selectedExam();
    const rows = [...(state.roster?.roster || [])]
        .filter(row => Number(row.sample_count || 0) > 0)
        .sort((a, b) => statusPriority(a.exam_status) - statusPriority(b.exam_status)
            || String(a.name || '').localeCompare(String(b.name || ''), 'tr-TR'));

    if (!rows.length) {
        grid.innerHTML = `<div class="feed-empty-state">${escapeHtml(localizeText('No enrolled students in this exam roster.'))}</div>`;
        return;
    }

    grid.innerHTML = rows.map(row => {
        const attempt = row.latest_attempt;
        const score = attempt ? `${(attempt.score * 100).toFixed(1)}%` : '-';
        const student = campusStudentById(row.student_id);
        const submittedPhoto = simulationSubmittedPhoto(row, student);
        const savedPhoto = row.reference_preview || student?.reference_preview;
        const progress = attempt ? Math.max(4, Math.min(100, Number(attempt.score || 0) * 100)) : 6;
        const sourceLabel = attempt
            ? attemptSourceLabel(attempt.attempt_source)
            : (submittedPhoto ? localizeText('Preloaded selfie') : localizeText('Awaiting selfie'));
        const selectedClass = row.student_id === state.simulationSelectedStudentId ? 'selected' : '';
        const cardLabel = `${row.name || row.student_id}, ${statusLabel(row.exam_status)}, ${attempt ? score : localizeText('no attempt yet')}`;
        return `
            <article class="student-feed-card ${statusClass(row.exam_status)} ${selectedClass}"
                role="button"
                tabindex="0"
                aria-label="${escapeHtml(cardLabel)}"
                data-simulation-student-card
                data-simulation-student="${escapeHtml(row.student_id)}"
                ${attempt ? `data-simulation-attempt="${escapeHtml(attempt.attempt_id)}"` : ''}>
                <div class="student-feed-top">
                    ${avatarHtml(savedPhoto)}
                    <div>
                        <strong>${escapeHtml(row.name || '-')}</strong>
                        <span>${escapeHtml(displayStudentId(row.student_id))}</span>
                    </div>
                    <i aria-hidden="true"></i>
                </div>
                <div class="student-feed-photos">
                    ${feedImageSlot('Submitted', submittedPhoto, sourceLabel)}
                    ${feedImageSlot('Saved', savedPhoto, `${Number(row.sample_count || 0)} ${localizeText('samples')}`)}
                </div>
                <div class="student-feed-score">
                    <span class="badge ${statusClass(row.exam_status)}">${escapeHtml(statusLabel(row.exam_status))}</span>
                    <span>${escapeHtml(score)}</span>
                    <div class="student-feed-scorebar" aria-hidden="true"><b style="width: ${progress.toFixed(1)}%"></b></div>
                </div>
            </article>
        `;
    }).join('');
}

function openSimulationFeedCard(card) {
    const attemptId = card.dataset.simulationAttempt;
    if (attemptId) {
        openSimulationAttempt(attemptId, card.dataset.simulationStudent);
        return;
    }
    setSelectValue('student-select', card.dataset.simulationStudent);
    setSelectValue('simulation-student', card.dataset.simulationStudent);
    state.selectedAttempt = null;
    state.selectedReviewStudent = null;
    state.attemptHistory = [];
    state.simulationSelectedAttemptId = SIMULATION_NO_ATTEMPT;
    syncSimulationSelection();
    renderSimulation();
}

function simulationSubmittedPhoto(row, student) {
    if (row?.latest_attempt?.query_preview) return row.latest_attempt.query_preview;
    if (row?.face_source === 'flux_synid' || student?.face_source === 'flux_synid') {
        return api.studentFluxTestImageUrl(row.student_id);
    }
    return null;
}

function feedImageSlot(label, src, detail) {
    const title = `${localizeText(label)}${detail ? `: ${detail}` : ''}`;
    return `
        <figure class="feed-photo ${src ? 'has-image' : 'missing'}" title="${escapeHtml(title)}">
            <div class="feed-photo-frame">
                <span class="feed-photo-label">${escapeHtml(label === 'Submitted' ? 'SUB' : 'SAVED')}</span>
                ${src ? `<img src="${escapeHtml(src)}" alt="" loading="lazy" onerror="this.closest('.feed-photo').classList.add('missing'); this.remove();">` : ''}
                <span class="feed-photo-missing">${escapeHtml(localizeText('No image'))}</span>
            </div>
        </figure>
    `;
}

async function openSimulationAttempt(attemptId, studentId) {
    if (studentId) {
        setSelectValue('student-select', studentId);
        setSelectValue('simulation-student', studentId);
        syncSimulationSelection();
    }
    state.simulationSelectedAttemptId = attemptId;
    await selectAttempt(attemptId, { scroll: false, syncSimulationAttempt: true });
    renderSimulation();
}

function renderSimulationTimeline() {
    const list = document.getElementById('simulation-timeline-list');
    const rows = [...(state.auditRows || [])].reverse().slice(0, 8);
    if (!rows.length) {
        list.innerHTML = `<li class="empty-timeline">${escapeHtml(localizeText('No campus events yet.'))}</li>`;
        return;
    }
    list.innerHTML = rows.map(row => `
        <li>
            <span>${escapeHtml(formatDateTime(row.timestamp))}</span>
            <strong>${escapeHtml(eventLabel(row.event_type || 'event'))}</strong>
            <p>${escapeHtml(localizeText(row.message || '-'))}</p>
        </li>
    `).join('');
}

function renderAuditRail(containerId, attempt, row = null) {
    const host = document.getElementById(containerId);
    if (!host) return;
    if (!attempt) {
        host.innerHTML = `<div class="audit-empty-state">${escapeHtml(localizeText('No verification attempt selected yet.'))}</div>`;
        return;
    }

    const stages = auditRailStages(attempt, row);
    host.innerHTML = stages.map((stage, index) => `
        <div class="audit-stage ${stage.kind}">
            <span class="audit-stage-icon" aria-hidden="true">${iconSvg(stage.icon)}</span>
            <div>
                <time>${escapeHtml(stage.time)}</time>
                <strong>${escapeHtml(stage.title)}</strong>
                <small>${escapeHtml(stage.detail)}</small>
            </div>
            ${index < stages.length - 1 ? '<i aria-hidden="true"></i>' : ''}
        </div>
    `).join('');
}

function auditRailStages(attempt, row) {
    const time = formatAuditTime(attempt.timestamp);
    const score = Number.isFinite(Number(attempt.score))
        ? `${(Number(attempt.score) * 100).toFixed(1)}%`
        : '-';
    const warnings = attempt.warnings || [];
    const finalStatus = attempt.final_status || row?.exam_status || attempt.status || 'Pending Verification';
    const livenessDetail = warnings.length ? localizeText('Warnings present') : localizeText('Passed');
    return [
        { icon: 'play', title: localizeText('Attempt Started'), detail: displayStudentId(attempt.student_id), time, kind: 'started' },
        { icon: 'camera', title: localizeText('Photo Captured'), detail: attemptSourceLabel(attempt.attempt_source), time, kind: 'captured' },
        { icon: 'shield', title: localizeText('Liveness Check'), detail: livenessDetail, time, kind: warnings.length ? 'review' : 'passed' },
        { icon: 'scan', title: localizeText('Face Analyzed'), detail: modelLabel(attempt.model_type), time, kind: 'analyzed' },
        { icon: 'percent', title: localizeText(`Match Score: ${score}`), detail: scoreDetail(attempt), time, kind: resultClass(attempt.decision) },
        { icon: finalStatusIcon(finalStatus), title: statusLabel(finalStatus), detail: finalStatusDetail(finalStatus), time, kind: resultClass(attempt.decision) },
    ];
}

function iconSvg(name) {
    const icons = {
        chevron: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M9 18l6-6-6-6"></path></svg>',
        play: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M8 5v14l11-7-11-7z"></path></svg>',
        camera: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M4 8h4l2-3h4l2 3h4v11H4V8z"></path><circle cx="12" cy="13" r="3"></circle></svg>',
        shield: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M12 3l7 3v5c0 4.4-2.8 8.2-7 10-4.2-1.8-7-5.6-7-10V6l7-3z"></path><path d="M9 12l2 2 4-5"></path></svg>',
        scan: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3"></path><path d="M8 12h8"></path><path d="M12 8v8"></path></svg>',
        percent: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M19 5L5 19"></path><circle cx="7" cy="7" r="2"></circle><circle cx="17" cy="17" r="2"></circle></svg>',
        granted: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="9"></circle><path d="M8 12l2.5 2.5L16 9"></path></svg>',
        review: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M12 8v5"></path><path d="M12 17h.01"></path><path d="M10.3 4.3h3.4L21 17.2 19.3 20H4.7L3 17.2 10.3 4.3z"></path></svg>',
        denied: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="9"></circle><path d="M15 9l-6 6M9 9l6 6"></path></svg>',
        waiting: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="9"></circle><path d="M12 7v5l3 2"></path></svg>',
    };
    return icons[name] || icons.waiting;
}

function attemptSourceLabel(source) {
    if (source === 'preloaded_demo') return localizeText('Preloaded demo selfie');
    if (source === 'upload') return localizeText('Uploaded selfie');
    return localizeText('Uploaded selfie');
}

function scoreDetail(attempt) {
    const score = Number(attempt.score);
    const threshold = Number(attempt.threshold);
    if (!Number.isFinite(score) || !Number.isFinite(threshold)) return localizeText('Awaiting threshold');
    if (score >= threshold) return localizeText('At or above threshold');
    return localizeText('Below threshold');
}

function finalStatusIcon(status) {
    if (ACCESS_GRANTED_STATUSES.has(status)) return 'granted';
    if (REVIEW_STATUSES.has(status)) return 'review';
    if (BLOCKED_STATUSES.has(status)) return 'denied';
    return 'waiting';
}

function finalStatusDetail(status) {
    if (ACCESS_GRANTED_STATUSES.has(status)) return localizeText('Access can proceed');
    if (REVIEW_STATUSES.has(status)) return localizeText('Auto-flagged');
    if (BLOCKED_STATUSES.has(status)) return localizeText('Access denied');
    return localizeText('Awaiting action');
}

function formatAuditTime(timestamp) {
    if (!timestamp) return '-';
    const parsed = new Date(timestamp);
    if (Number.isNaN(parsed.getTime())) return String(timestamp).split('T').pop() || timestamp;
    return formatTime(parsed, { second: '2-digit' });
}

function renderSimulationPaneSwitch() {
    const activePane = state.simulationMobilePane || 'student';
    document.querySelectorAll('[data-simulation-pane]').forEach(button => {
        const isActive = button.dataset.simulationPane === activePane;
        button.classList.toggle('active', isActive);
        button.setAttribute('aria-selected', isActive ? 'true' : 'false');
    });
    const layoutRoot = document.getElementById('simulation-layout-root');
    if (layoutRoot) {
        layoutRoot.setAttribute('data-active-pane', activePane);
    }
    document.querySelector('.student-pov')?.classList.toggle('mobile-active', activePane === 'student');
    document.querySelector('.instructor-pov')?.classList.toggle('mobile-active', activePane === 'instructor');
}

function simulationAttempt() {
    const row = currentRosterRow();
    if (state.simulationSelectedAttemptId === SIMULATION_NO_ATTEMPT) return null;
    if (
        state.selectedAttempt?.attempt_id === state.simulationSelectedAttemptId
        && attemptMatchesSimulationContext(state.selectedAttempt)
    ) {
        return state.selectedAttempt;
    }
    if (row?.latest_attempt?.attempt_id === state.simulationSelectedAttemptId) return row.latest_attempt;
    return state.simulationSelectedAttemptId ? null : row?.latest_attempt || null;
}

function attemptMatchesSimulationContext(attempt) {
    const row = currentRosterRow();
    const examId = currentExamId();
    return Boolean(
        attempt
        && row?.student_id === attempt.student_id
        && (!examId || attempt.exam_id === examId)
    );
}

function syncSimulationAttemptSelection(attempt) {
    if (attemptMatchesSimulationContext(attempt)) {
        state.simulationSelectedAttemptId = attempt.attempt_id;
    }
}

function simulationStudentGuidance(context, attempt) {
    if (!context.student) return localizeText('Select a student to start the split-screen simulation.');
    if (context.modelMismatch) return modelMismatchMessage(context);
    if (context.sampleCount <= 0) return localizeText('Enrollment is missing. The student cannot verify until face samples are enrolled.');
    if (context.sampleCount < ENROLLMENT_TARGET) return enrollmentGuidanceMessage(context);
    if (!document.getElementById('simulation-consent-check').checked) return localizeText('Consent is required before the exam-day selfie can be submitted.');
    if (!attempt) return localizeText('Enrollment is ready. Submit an exam-day selfie to see the instructor dashboard update.');
    return statusConsequence(attempt.final_status || attempt.status);
}

function studentFacingTitle(attempt) {
    if (attempt.final_status === 'Approved by Proctor') return statusLabel('Approved by Proctor');
    if (attempt.final_status === 'Verified' || attempt.decision === 'verified') return localizeText('Model verified');
    if (REVIEW_STATUSES.has(attempt.final_status)) return localizeText('Manual review required');
    if (BLOCKED_STATUSES.has(attempt.final_status)) return localizeText('Access blocked');
    return humanizeDecision(attempt.decision);
}

function studentFacingMessage(attempt) {
    if (attempt.final_status === 'Approved by Proctor') return localizeText('A proctor approved this access decision.');
    if (attempt.final_status === 'Verified' || attempt.decision === 'verified') return localizeText('Model verified. Access can proceed if the exam policy allows it.');
    if (attempt.final_status === 'Fallback Requested') return localizeText('A proctor has requested a fallback ID check before exam access.');
    if (attempt.decision === 'manual_review') return localizeText('A proctor must review this attempt before access is granted.');
    return localizeText('This attempt did not meet the exam access policy.');
}

function setSimulationMiniStep(id, done, active) {
    const item = document.getElementById(id);
    item.classList.toggle('done', done);
    item.classList.toggle('active', active);
}

function setSimulationBadge(id, text, extraClass = '') {
    const badge = document.getElementById(id);
    badge.className = `badge ${statusClass(text)} ${extraClass}`.trim();
    badge.textContent = statusLabel(text || '-');
}

async function verifySimulationStudent() {
    const examId = currentExamId();
    const studentId = valueOf('simulation-student');
    const file = document.getElementById('simulation-verify-file').files[0];
    if (!document.getElementById('simulation-consent-check').checked) {
        toast('Consent is required before camera or upload verification.', 'error');
        return;
    }
    if (!examId || !studentId) {
        toast('Select an exam and student first.', 'error');
        return;
    }
    if (!file && !isCurrentStagedPreloaded('simulation', examId, studentId)) {
        toast('Select an exam, student, and exam-day selfie.', 'error');
        return;
    }
    if (!confirmVerification(examId, studentId, file ? 'upload' : 'preloaded_demo', file?.name)) return;
    setTextIfPresent('simulation-verify-error', '');
    try {
        setVerificationInFlight(true);
        setSelectValue('student-select', studentId);
        const result = file
            ? await api.verifyStudent(examId, studentId, file)
            : await api.verifyPreloadedStudent(examId, studentId, state.stagedPreloadedSelfie.scenario, true);
        state.simulationSelectedAttemptId = result.attempt.attempt_id;
        state.selectedAttempt = result.attempt;
        renderStudentResult(result);
        document.getElementById('simulation-verify-file').value = '';
        clearStagedPreloadedSelfie();
        await refreshAll();
        await selectAttempt(result.attempt.attempt_id, { scroll: false });
    } catch (err) {
        const message = apiErrorMessage(err);
        setTextIfPresent('simulation-student-guidance', message);
        setTextIfPresent('simulation-verify-error', message);
        toast(err, 'error');
        renderSimulation();
    } finally {
        setVerificationInFlight(false);
    }
}

async function reviewSimulationAttempt(action) {
    const attempt = simulationAttempt();
    if (!attempt) {
        toast('Submit or select an attempt before reviewing.', 'error');
        return;
    }
    if (!isReviewActionAllowed(attempt)) {
        toast('This attempt is not eligible for manual review.', 'error');
        return;
    }
    try {
        await api.reviewAttempt(
            attempt.attempt_id,
            valueOf('simulation-reviewer-name'),
            action,
            valueOf('simulation-review-reason')
        );
        state.simulationSelectedAttemptId = attempt.attempt_id;
        toast('Simulation review saved.', 'success');
        await refreshAll();
        await selectAttempt(attempt.attempt_id, { scroll: false, syncSimulationAttempt: true });
    } catch (err) {
        toast(err, 'error');
    }
}

async function compareInLab() {
    const modelType = valueOf('lab-model');
    const imageA = document.getElementById('lab-file-a').files[0];
    const imageB = document.getElementById('lab-file-b').files[0];
    if (!imageA || !imageB) {
        toast('Choose two face images for model comparison.', 'error');
        return;
    }
    try {
        const result = await api.modelLabCompare(modelType, imageA, imageB);
        document.getElementById('lab-title').textContent = result.match ? localizeText('Likely Match') : localizeText('Likely Different');
        document.getElementById('lab-score').textContent = result.score.toFixed(3);
        document.getElementById('lab-threshold').textContent = result.threshold.toFixed(3);
        document.getElementById('lab-decision').textContent = result.match ? decisionLabel('Pass') : decisionLabel('Review');
        updateScoreBar('lab', result.score, result.threshold, result.match ? 'verified' : 'rejected');
    } catch (err) {
        toast(err, 'error');
    }
}

function updateScoreBar(prefix, score, threshold, decision) {
    const fill = document.getElementById(`${prefix}-score-fill`);
    const line = document.getElementById(`${prefix}-threshold-line`);
    fill.classList.remove('verified', 'review', 'rejected');
    if (decision === 'verified') fill.classList.add('verified');
    if (decision === 'manual_review') fill.classList.add('review');
    if (decision === 'rejected') fill.classList.add('rejected');
    fill.style.width = `${Math.max(0, Math.min(1, score)) * 100}%`;
    line.style.left = `${Math.max(0, Math.min(1, threshold)) * 100}%`;
    line.dataset.threshold = Number.isFinite(Number(threshold)) ? Number(threshold).toFixed(3) : '';
}

function requireConsent() {
    if (!document.getElementById('consent-check').checked) {
        toast('Consent is required before camera or upload verification.', 'error');
        return false;
    }
    return true;
}

function updateStudentProgress() {
    const context = enrollmentContext();
    const consentDone = document.getElementById('consent-check').checked;
    const enrollmentDone = !context.modelMismatch && context.sampleCount >= ENROLLMENT_TARGET;
    const resultDone = Boolean(state.studentResultAttemptId);

    setStepState('step-consent', consentDone, state.studentStep === 'consent');
    setStepState('step-enrollment', enrollmentDone, state.studentStep === 'enrollment');
    setStepState('step-verification', resultDone, state.studentStep === 'verification');
    setStepState('step-result', resultDone, state.studentStep === 'result');
    const gateAttempt = context.rosterRow?.latest_attempt
        || (state.selectedAttempt?.student_id === context.student?.student_id ? state.selectedAttempt : null);
    renderGateHeader(
        'gate',
        `${displayExamName(context.exam?.name)} access gate`,
        gateStatusInfo(context, gateAttempt),
        currentStudentGateStep()
    );
    renderStudentStepPanels();
}

function setStepState(id, done, active) {
    const item = document.getElementById(id);
    item.classList.toggle('done', done);
    item.classList.toggle('active', active);
}

function setStudentStep(step) {
    const targetStep = step === 'result' && !state.studentResultAttemptId ? 'verification' : step;
    state.studentStep = targetStep;
    renderStudentStepPanels();
    updateStudentProgress();
}

function currentStudentGateStep() {
    const index = STUDENT_STEP_ORDER.indexOf(state.studentStep);
    return index >= 0 ? index + 1 : 1;
}

function simulationGateStep(consentDone, enrollmentDone, verificationDone) {
    if (verificationDone) return 4;
    if (enrollmentDone) return 3;
    if (consentDone) return 2;
    return 1;
}

function gateStatusInfo(context, attempt) {
    if (attempt?.final_status || attempt?.status) {
        const status = attempt.final_status || attempt.status;
        return { label: status, className: gateStatusClass(status, attempt.decision) };
    }
    if (!context.student) return { label: 'Waiting', className: 'pending' };
    if (context.modelMismatch) return { label: 'Model Mismatch', className: 'review' };
    if (context.sampleCount <= 0) return { label: 'No Enrollment', className: 'pending' };
    if (context.sampleCount < ENROLLMENT_TARGET) return { label: 'Partial Enrollment', className: 'review' };
    return { label: 'Ready', className: 'verified' };
}

function gateStatusClass(status, decision = '') {
    if (ACCESS_GRANTED_STATUSES.has(status) || decision === 'verified') return 'verified';
    if (REVIEW_STATUSES.has(status) || decision === 'manual_review') return 'review';
    if (BLOCKED_STATUSES.has(status) || decision === 'rejected') return 'rejected';
    return 'pending';
}

function renderGateHeader(prefix, title, status, stepNumber) {
    const titleId = prefix === 'simulation' ? 'simulation-gate-title' : 'gate-step-title';
    const pillId = prefix === 'simulation' ? 'simulation-gate-status-pill' : 'gate-status-pill';
    const stepId = prefix === 'simulation' ? 'simulation-gate-step-pill' : 'gate-step-pill';
    setTextIfPresent(titleId, title || 'Exam access gate');
    setTextIfPresent(stepId, t('gate.step', { step: Math.max(1, Math.min(4, stepNumber || 1)) }));
    const pill = document.getElementById(pillId);
    if (!pill) return;
    pill.textContent = statusLabel(status?.label || 'Waiting');
    pill.className = `gate-status-pill ${status?.className || 'pending'}`;
}

function renderStudentStepPanels() {
    document.querySelectorAll('[data-student-step-panel]').forEach(panel => {
        panel.classList.toggle('active', panel.dataset.studentStepPanel === state.studentStep);
    });
}

function updateContextBar() {
    const exam = selectedExam();
    const course = selectedCourse();
    const student = selectedStudent();
    const threshold = Number(exam?.threshold);
    const thresholdText = Number.isFinite(threshold) ? threshold.toFixed(3) : '-';
    const courseName = displayCourseName(course?.name || state.roster?.course?.name);
    const examName = displayExamName(exam?.name);
    const scenarioName = `${courseCode(courseName)} ${examShortName(examName)}`.trim();
    const contextSummary = t('context.summary', {
        university: DISPLAY_UNIVERSITY,
        scenario: scenarioName || 'SE 204 Ara Sınav',
        model: modelLabel(exam?.model_type || '-'),
        threshold: thresholdText,
    });
    setTextIfPresent('context-summary', contextSummary);
    const topContextPill = document.querySelector('.top-context-pill');
    if (topContextPill) {
        topContextPill.textContent = `${scenarioName || 'SE 204 Ara Sınav'} · ${modelLabel(exam?.model_type || '-')} · ${thresholdText}`;
    }
    setTextIfPresent('context-university', DISPLAY_UNIVERSITY);
    setTextIfPresent('context-course', courseName || DISPLAY_COURSE);
    setTextIfPresent('context-exam', examName || DISPLAY_EXAM);
    setTextIfPresent('context-model', modelLabel(exam?.model_type || '-'));
    setTextIfPresent('context-threshold', thresholdText);
    setTextIfPresent('context-window', formatExamWindow(exam));
    setTextIfPresent('context-persona', PERSONAS[state.activeView] || PERSONAS.admin);

    const hybridBadge = document.getElementById('context-hybrid-badge');
    if (hybridBadge) {
        hybridBadge.classList.toggle('hidden', exam?.model_type !== 'hybrid');
    }

    renderPolicySummaries();
    renderGateContextPanel();
}

function renderPolicySummaries() {
    const exam = selectedExam();
    const model = modelLabel(exam?.model_type || '-');
    const threshold = Number(exam?.threshold);
    const thresholdText = Number.isFinite(threshold) ? threshold.toFixed(3) : '-';
    const title = t('policy.title', { model });
    const detail = t('policy.detail', { threshold: thresholdText });
    setTextIfPresent('student-policy-title', title);
    setTextIfPresent('student-policy-detail', detail);
    setTextIfPresent('student-policy-model', model);
    setTextIfPresent('gate-policy-title', title);
    setTextIfPresent('gate-policy-detail', detail);
    setTextIfPresent('gate-policy-model', model);
    setTextIfPresent('admin-policy-title', title);
    setTextIfPresent('admin-policy-detail', `${detail} ${t('policy.adminDetail')}`);
    setTextIfPresent('admin-policy-model', model);
}

function renderGateContextPanel() {
    if (!document.getElementById('gate-selected-student')) return;
    const context = enrollmentContext();
    const row = context.rosterRow;
    const attempt = row?.latest_attempt;
    const studentText = context.student
        ? `${context.student.name} - ${displayStudentId(context.student.student_id)}`
        : localizeText('Select a student');

    let enrollmentText = '-';
    if (!context.student) {
        enrollmentText = localizeText('Select a student');
    } else if (context.modelMismatch) {
        enrollmentText = modelMismatchMessage(context);
    } else if (context.sampleCount <= 0) {
        enrollmentText = localizeText('Enrollment needed before verification');
    } else if (context.sampleCount < ENROLLMENT_TARGET) {
        enrollmentText = localizeText(`${context.sampleCount}/${ENROLLMENT_TARGET} samples; usable, but not ideal`);
    } else {
        enrollmentText = localizeText(`Ready with ${context.sampleCount} sample(s)`);
    }

    const lastAttemptText = attempt
        ? `${statusLabel(attempt.final_status || row?.exam_status || 'Attempt recorded')} | skor ${numberOrDash(attempt.score)} | ${formatDateTime(attempt.timestamp)}`
        : statusConsequence(row?.exam_status || 'Not Started');

    setTextIfPresent('gate-selected-student', studentText);
    setTextIfPresent('gate-enrollment-state', enrollmentText);
    setTextIfPresent('gate-last-attempt', lastAttemptText);
    setAvatar('gate-student-avatar', row?.reference_preview || context.student?.reference_preview);
    const testImageLink = document.getElementById('gate-download-test-image');
    if (testImageLink) {
        const canDownload = context.student?.face_source === 'flux_synid';
        testImageLink.classList.toggle('hidden', !canDownload);
        testImageLink.href = canDownload
            ? api.studentFluxTestImageUrl(context.student.student_id)
            : '#';
    }
}

function normalizeViewName(viewName) {
    if (viewName === 'evidence' || viewName === 'lab') return 'admin';
    return viewName;
}

function defaultNavTarget(viewName, normalizedView = normalizeViewName(viewName)) {
    if (viewName === 'evidence' || viewName === 'lab') return 'audit-logs';
    return normalizedView || viewName || 'simulation';
}

function updateScenarioRail(viewName) {
    const activeStep = viewName === 'lab' ? 'evidence' : viewName;
    document.querySelectorAll('.scenario-step').forEach(button => {
        button.classList.toggle('active', button.dataset.demoJump === activeStep);
    });
}

function openOperationPanel(id) {
    const panel = document.getElementById(id);
    if (!panel) return;
    panel.open = true;
    requestAnimationFrame(() => panel.scrollIntoView({ behavior: 'smooth', block: 'start' }));
}

function demoStudentId() {
    const aylin = state.snapshot?.students?.find(student => student.name === 'Aylin Kaya');
    return aylin?.student_id || FALLBACK_DEMO_STUDENT_ID;
}

function demoExamId() {
    const preferred = state.snapshot?.exams?.find(exam => exam.exam_id === FALLBACK_DEMO_EXAM_ID);
    return preferred?.exam_id || state.snapshot?.exams?.[0]?.exam_id || FALLBACK_DEMO_EXAM_ID;
}

function displayCourseName(name) {
    if (!name) return DISPLAY_COURSE;
    return String(name)
        .replace('CS 204', 'SE 204')
        .replace('Data Structures', 'Veri Yapıları');
}

function displayExamName(name) {
    if (!name || name === 'Midterm 1') return DISPLAY_EXAM;
    return String(name).replace('Midterm Exam', DISPLAY_EXAM).replace('Midterm 1', DISPLAY_EXAM);
}

function displayInstructor(name) {
    if (!name || name === 'Dr. Elena Morris') return 'Dr. Selin Demir';
    return String(name);
}

function displayTerm(term) {
    if (!term) return 'Demo Dönemi';
    return String(term).replace('Spring', 'Bahar').replace('Demo Term', 'Demo Dönemi');
}

function displayStudentId(studentId) {
    if (!studentId || studentId === FALLBACK_DEMO_STUDENT_ID) return DISPLAY_STUDENT_ID;
    return String(studentId).replace(/^NB-/, 'AT-');
}

function courseCode(courseName) {
    return String(courseName || DISPLAY_COURSE).split(' - ')[0];
}

function examShortName(examName) {
    return String(examName || DISPLAY_EXAM).replace(' Exam', '');
}

function currentExamId() {
    return document.getElementById('student-exam').value || state.snapshot?.exams?.[0]?.exam_id;
}

function selectedExam() {
    const examId = currentExamId();
    return state.snapshot?.exams?.find(exam => exam.exam_id === examId);
}

function selectedCourse() {
    const exam = selectedExam();
    return state.snapshot?.courses?.find(course => course.course_id === exam?.course_id) || state.snapshot?.courses?.[0];
}

function selectedStudent() {
    const studentId = document.getElementById('student-select').value;
    return campusStudentById(studentId);
}

function campusStudentById(studentId) {
    return state.snapshot?.students?.find(student => student.student_id === studentId);
}

function currentRosterRow() {
    const studentId = document.getElementById('student-select').value;
    return state.roster?.roster?.find(row => row.student_id === studentId);
}

function selectedCampusEnrollment() {
    const studentId = document.getElementById('student-select').value;
    if (!studentId) return null;
    return state.enrollments.find(enrollment => enrollment.user_id === `campus_${studentId}`) || null;
}

function valueOf(id) {
    return document.getElementById(id).value.trim();
}

function setSelectValue(id, value) {
    const select = document.getElementById(id);
    if ([...select.options].some(option => option.value === value)) {
        select.value = value;
    }
}

function toDatetimeLocal(value) {
    return value ? value.slice(0, 16) : '';
}

function formatExamWindow(exam) {
    if (!exam?.start_time || !exam?.end_time) return '-';
    const start = new Date(exam.start_time);
    const end = new Date(exam.end_time);
    if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime())) {
        return `${exam.start_time.slice(0, 16).replace('T', ' ')} - ${exam.end_time.slice(11, 16)}`;
    }
    return `${formatDateTime(start)} - ${formatTime(end)}`;
}

function statusClass(status) {
    return String(status || '').toLowerCase().replaceAll(' ', '-');
}

function resultClass(decision) {
    if (decision === 'verified') return 'verified';
    if (decision === 'manual_review') return 'review';
    return 'rejected';
}

function modelLabel(modelType) {
    const labels = {
        siamese: 'Siamese',
        prototypical: 'Prototypical',
        hybrid: 'Hybrid FaceNet',
        facenet_proto: 'FaceNet Proto',
        facenet_contrastive_proto: 'FaceNet Contrastive Proto',
        facenet_contrastive_proto_model5: 'FaceNet Contrastive Proto Model 5',
        facenet_arcface_triplet_model6: 'FaceNet ArcFace Triplet Model 6'
    };
    return labels[modelType] || String(modelType || '-');
}

function humanizeDecision(decision) {
    return decisionLabel(decision);
}

function statusConsequence(status) {
    if (ACCESS_GRANTED_STATUSES.has(status)) return localizeText('Access granted');
    if (REVIEW_STATUSES.has(status)) return localizeText('Proctor action needed');
    if (BLOCKED_STATUSES.has(status)) return localizeText('Exam access blocked');
    if (status === 'Enrollment Needed') return localizeText('Enroll before verification');
    if (status === 'Not Started') return localizeText('Waiting for student');
    if (status === 'Pending Verification') return localizeText('Verification in progress');
    return localizeText('Waiting');
}

function numberOrDash(value) {
    return Number.isFinite(Number(value)) ? Number(value).toFixed(3) : '-';
}

function setTextIfPresent(id, text) {
    const element = document.getElementById(id);
    if (element) element.textContent = localizeText(text);
}

function escapeHtml(value) {
    return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
}

function toast(message, type = 'info') {
    const host = document.getElementById('toast-host');
    const item = document.createElement('div');
    item.className = `toast ${type}`;
    item.textContent = apiErrorMessage(message);
    host.appendChild(item);
    setTimeout(() => item.remove(), 4200);
}
