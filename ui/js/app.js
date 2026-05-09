const state = {
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
    studentResultAttemptId: null,
    studentStep: 'consent',
    activeView: 'simulation',
    simulationScenario: 'matching',
    simulationMobilePane: 'student',
    simulationSelectedStudentId: null,
    simulationSelectedAttemptId: null
};

const ENROLLMENT_TARGET = 3;
const ENROLLMENT_MAX = 5;
const DISPLAY_UNIVERSITY = 'Atılım University Demo';
const DISPLAY_COURSE = 'SE 204 - Data Structures';
const DISPLAY_EXAM = 'Midterm Exam';
const DISPLAY_STUDENT_ID = 'AT-2026-1042';
const FALLBACK_DEMO_STUDENT_ID = 'NB-2026-1042';
const FALLBACK_DEMO_EXAM_ID = 'CS204-MIDTERM-1';
const REVIEW_STATUSES = new Set(['Manual Review', 'Fallback Requested']);
const ACCESS_GRANTED_STATUSES = new Set(['Verified', 'Approved by Proctor']);
const BLOCKED_STATUSES = new Set(['Rejected']);
const WAITING_STATUSES = new Set(['Not Started', 'Enrollment Needed', 'Pending Verification']);
const STUDENT_STEP_ORDER = ['consent', 'enrollment', 'verification', 'result'];
const KPI_KEYS = ['attempts', 'verified', 'review', 'blocked', 'no-enrollment'];
const PERSONAS = {
    launch: 'Demo Operator',
    simulation: 'Live Simulation',
    student: 'Student: Aylin Kaya',
    proctor: 'Review Desk: Proctor Lee',
    admin: 'Operations: Registrar Admin',
    evidence: 'Operations: Model Evidence'
};

document.addEventListener('DOMContentLoaded', () => {
    bindTabs();
    bindActions();
    updateSidebarClock();
    window.setInterval(updateSidebarClock, 60000);
    showView('simulation');
    refreshAll();
    setStudentStep('consent');
});

function bindTabs() {
    document.querySelectorAll('.tab-button').forEach(button => {
        button.addEventListener('click', () => showView(button.dataset.view));
    });
}

function showView(viewName) {
    const normalizedView = normalizeViewName(viewName);
    state.activeView = viewName === 'evidence' ? 'evidence' : normalizedView;
    document.querySelectorAll('.tab-button').forEach(button => {
        button.classList.toggle('active', button.dataset.view === normalizedView);
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
    if (viewName === 'evidence' || viewName === 'lab') {
        openOperationPanel('operation-model-evidence');
    }
}

function bindActions() {
    document.getElementById('launch-exam-btn').addEventListener('click', () => {
        showView('admin');
        setSelectValue('student-select', demoStudentId());
        renderEnrollmentGuidance();
    });
    document.getElementById('guide-jump-btn').addEventListener('click', () => {
        jumpFromGuide(valueOf('guide-jump-select'));
    });
    document.querySelectorAll('[data-demo-jump]').forEach(button => {
        button.addEventListener('click', () => jumpFromGuide(button.dataset.demoJump));
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
    document.getElementById('wrong-face-demo-btn').addEventListener('click', () => showView('proctor'));
    document.getElementById('save-course-btn').addEventListener('click', saveCourse);
    document.getElementById('save-exam-btn').addEventListener('click', saveExam);
    document.getElementById('import-roster-btn').addEventListener('click', importRoster);
    document.getElementById('reset-demo-btn').addEventListener('click', resetDemo);
    document.getElementById('flux-preupload-btn').addEventListener('click', preuploadFlux);
    document.getElementById('flux-export-btn').addEventListener('click', exportFluxTestSet);
    document.getElementById('verify-preloaded-btn').addEventListener('click', () => verifyPreloadedStudent(false));
    document.getElementById('simulation-preloaded-btn').addEventListener('click', () => verifyPreloadedStudent(true));
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
    document.getElementById('simulation-feed-table').addEventListener('click', event => {
        const button = event.target.closest('[data-simulation-attempt]');
        if (!button) return;
        openSimulationAttempt(button.dataset.simulationAttempt, button.dataset.simulationStudent);
    });
    document.getElementById('student-exam').addEventListener('change', async () => {
        resetStudentResultPanel();
        await refreshRosterOnly();
        syncAdminFields();
        updateContextBar();
        renderEnrollmentGuidance();
        syncSimulationControls();
        renderSimulation();
    });
    document.getElementById('student-select').addEventListener('change', () => {
        resetStudentResultPanel();
        updateContextBar();
        renderEnrollmentGuidance();
        syncSimulationSelection();
        renderSimulation();
    });
    document.getElementById('simulation-exam').addEventListener('change', async () => {
        setSelectValue('student-exam', valueOf('simulation-exam'));
        resetStudentResultPanel();
        await refreshRosterOnly();
        syncAdminFields();
        updateContextBar();
        renderEnrollmentGuidance();
        renderSimulation();
    });
    document.getElementById('simulation-student').addEventListener('change', () => {
        setSelectValue('student-select', valueOf('simulation-student'));
        resetStudentResultPanel();
        syncSimulationSelection();
        updateContextBar();
        renderEnrollmentGuidance();
        renderSimulation();
    });
    document.getElementById('simulation-consent-check').addEventListener('change', renderSimulation);
    document.getElementById('simulation-verify-file').addEventListener('change', renderSimulation);
    document.getElementById('enroll-files').addEventListener('change', renderEnrollmentFileState);
    document.getElementById('consent-check').addEventListener('change', updateStudentProgress);
    document.getElementById('verify-file').addEventListener('change', updateStudentProgress);
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
        toast(err.message, 'error');
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
        toast(err.message, 'error');
    }
}

function jumpFromGuide(viewName) {
    setSelectValue('student-select', demoStudentId());
    if (viewName === 'student') {
        document.getElementById('consent-check').checked = true;
        setStudentStep('enrollment');
        renderEnrollmentGuidance();
    }
    if (viewName === 'proctor') {
        setReviewFilter('needs_review');
    }
    if (viewName === 'simulation') {
        document.getElementById('simulation-consent-check').checked = true;
        syncSimulationSelection();
        renderSimulation();
    }
    showView(viewName);
}

function setStatus(text, ok) {
    const status = document.getElementById('system-status');
    document.getElementById('status-text').textContent = text;
    status.classList.toggle('offline', !ok);
    status.classList.toggle('online', ok);
}

function updateSidebarClock() {
    const now = new Date();
    setTextIfPresent('sidebar-clock-time', now.toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit'
    }));
    setTextIfPresent('sidebar-clock-date', now.toLocaleDateString([], {
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
        document.getElementById('course-term').value = course.term;
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

function renderMetrics() {
    const roster = state.roster?.roster || [];
    const metrics = dashboardMetrics(roster);

    document.getElementById('metric-students').textContent = roster.length;
    document.getElementById('metric-verified').textContent = metrics.verified;
    document.getElementById('metric-review').textContent = metrics.review;
    setTextIfPresent('sidebar-review-count', metrics.review);
    setTextIfPresent('sidebar-blocked-count', metrics.blocked);
    setTextIfPresent(
        'queue-summary',
        `${metrics.review} need review | ${metrics.verified} verified | ${metrics.waiting} waiting | ${metrics.blocked} blocked`
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
            key === 'attempts' ? 'Live' : `${((count / denominator) * 100).toFixed(1)}%`
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
    return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(key)} trend"><polyline points="${points}"></polyline></svg>`;
}

function renderRoster() {
    const body = document.getElementById('roster-table');
    const allRows = state.roster?.roster || [];
    const roster = filteredRosterRows(allRows);
    if (!allRows.length) {
        body.innerHTML = '<tr><td colspan="4" class="empty-cell">No roster rows for this exam.</td></tr>';
        return;
    }
    if (!roster.length) {
        body.innerHTML = '<tr><td colspan="4" class="empty-cell">No roster rows match these filters.</td></tr>';
        return;
    }
    body.innerHTML = roster.map(row => {
        const latest = row.latest_attempt;
        const score = latest ? latest.score.toFixed(3) : '-';
        const actionLabel = REVIEW_STATUSES.has(row.exam_status) ? 'Review' : 'Open';
        const action = latest
            ? `<button class="row-button ${REVIEW_STATUSES.has(row.exam_status) ? 'review-action' : ''}" data-attempt="${escapeHtml(latest.attempt_id)}">${actionLabel}</button>`
            : '<span class="muted action-waiting">Waiting</span>';
        const selected = latest?.attempt_id === state.selectedAttempt?.attempt_id ? ' class="selected-row"' : '';
        return `
            <tr${selected}>
                <td>${studentNameCell(row)}</td>
                <td><span class="badge ${statusClass(row.exam_status)}">${escapeHtml(row.exam_status)}</span></td>
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
    if (sort === 'name') return String(a.name || '').localeCompare(String(b.name || ''));
    if (sort === 'student_id') return String(a.student_id || '').localeCompare(String(b.student_id || ''));
    if (sort === 'score') {
        const aScore = Number.isFinite(Number(a.latest_attempt?.score)) ? Number(a.latest_attempt.score) : 2;
        const bScore = Number.isFinite(Number(b.latest_attempt?.score)) ? Number(b.latest_attempt.score) : 2;
        return aScore - bScore;
    }
    return statusPriority(a.exam_status) - statusPriority(b.exam_status)
        || String(a.name || '').localeCompare(String(b.name || ''));
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
        state.simulationSelectedAttemptId = attempt.attempt_id;
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
        toast(err.message, 'error');
    }
}

function renderReviewAttempt(attempt, student, history = []) {
    document.getElementById('review-panel').classList.remove('review-empty');
    document.getElementById('review-title').textContent =
        `${student.name || attempt.student_name || attempt.student_id} - ${attempt.final_status}`;
    document.getElementById('review-empty-state').textContent = 'Attempt loaded. Review the images and status before taking action.';
    document.getElementById('review-status').textContent = attempt.final_status || attempt.status || '-';
    document.getElementById('review-decision').textContent = humanizeDecision(attempt.decision);
    document.getElementById('review-model').textContent = modelLabel(attempt.model_type);
    document.getElementById('review-score').textContent = numberOrDash(attempt.score);
    document.getElementById('review-threshold').textContent = numberOrDash(attempt.threshold);
    document.getElementById('review-time').textContent = attempt.timestamp || '-';
    document.getElementById('review-warnings').textContent =
        attempt.warnings?.length ? attempt.warnings.join('; ') : 'None';
    document.getElementById('review-previous-attempts').textContent =
        history.length > 1 ? `${history.length - 1} earlier attempt(s) for this exam.` : 'No earlier attempts for this exam.';
    setPreview('reference-preview', student.reference_preview);
    setPreview('query-preview', attempt.query_preview);
    renderAuditRail('review-audit-rail', attempt, student);
    setReviewButtonsEnabled(true);
}

function setReviewLoading() {
    state.selectedAttempt = null;
    state.selectedReviewStudent = null;
    state.attemptHistory = [];
    document.getElementById('review-panel').classList.add('review-empty');
    document.getElementById('review-title').textContent = 'Loading attempt...';
    document.getElementById('review-empty-state').textContent = 'Loading attempt details and previews.';
    clearReviewDetails();
    setReviewButtonsEnabled(false);
}

function setReviewError(message) {
    state.selectedAttempt = null;
    state.selectedReviewStudent = null;
    state.attemptHistory = [];
    document.getElementById('review-panel').classList.add('review-empty');
    document.getElementById('review-title').textContent = 'Attempt could not be loaded';
    document.getElementById('review-empty-state').textContent = message || 'Attempt could not be loaded.';
    clearReviewDetails();
    setReviewButtonsEnabled(false);
}

function resetReviewPanel() {
    state.selectedAttempt = null;
    state.selectedReviewStudent = null;
    state.attemptHistory = [];
    document.getElementById('review-panel').classList.add('review-empty');
    document.getElementById('review-title').textContent = 'Select an attempt';
    document.getElementById('review-empty-state').textContent = 'Select an attempt from the Needs Review queue or roster.';
    clearReviewDetails();
    setReviewButtonsEnabled(false);
}

function clearReviewDetails() {
    ['review-status', 'review-decision', 'review-model', 'review-score', 'review-threshold', 'review-time', 'review-warnings', 'review-previous-attempts']
        .forEach(id => {
            document.getElementById(id).textContent = '-';
        });
    setPreview('reference-preview', null);
    setPreview('query-preview', null);
    renderAuditRail('review-audit-rail', null, null);
}

function setReviewButtonsEnabled(enabled) {
    document.querySelectorAll('[data-review-action]').forEach(button => {
        button.disabled = !enabled;
    });
}

function setPreview(id, dataUrl) {
    const host = document.getElementById(id);
    if (!host) return;
    host.innerHTML = dataUrl ? `<img src="${dataUrl}" alt="">` : 'No image';
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
    const modelType = selectedExam()?.model_type || 'siamese';
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
        const message = `You selected ${files.length} file(s), but only ${context.remaining} enrollment slot(s) remain.`;
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
        document.getElementById('enroll-warning').textContent = err.message;
        toast(err.message, 'error');
    }
}

function renderEnrollmentGuidance() {
    const context = enrollmentContext();
    const studentText = context.student
        ? `${context.student.name} - ${displayStudentId(context.student.student_id)}`
        : 'Select a student';

    renderPolicySummaries();
    document.getElementById('enrollment-student-label').textContent = studentText;
    document.getElementById('enrollment-model-label').textContent =
        `${modelLabel(context.exam?.model_type || '-')} enrollment for this exam`;
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
        ? `${files.length} file(s) selected. ${context.remaining} enrollment slot(s) left.`
        : 'No files selected.';

    if (context.modelMismatch) {
        warning.textContent = modelMismatchMessage(context);
    } else if (files.length > context.remaining) {
        warning.textContent = `You selected ${files.length} file(s), but only ${context.remaining} enrollment slot(s) remain.`;
    } else {
        warning.textContent = '';
    }

    button.textContent = files.length
        ? `Add ${files.length} Sample${files.length === 1 ? '' : 's'}`
        : 'Add Face Samples';
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
    if (!context.student) return 'Select a student to view enrollment guidance.';
    if (context.modelMismatch) return modelMismatchMessage(context);
    if (context.sampleCount <= 0) return 'Add 3 clear face samples before verification.';
    if (context.sampleCount < ENROLLMENT_TARGET) {
        const remainingRecommended = ENROLLMENT_TARGET - context.sampleCount;
        return `Enrollment works, but add ${remainingRecommended} more sample${remainingRecommended === 1 ? '' : 's'} for a stronger prototype.`;
    }
    return 'Enrollment ready. Continue to exam-day selfie.';
}

function enrollmentNextAction(context) {
    if (context.modelMismatch) return 'This model needs a separate enrollment before verification.';
    if (context.sampleCount <= 0) return 'Add 3 clear samples before submitting the exam-day selfie.';
    if (context.sampleCount < ENROLLMENT_TARGET) {
        return `Add ${ENROLLMENT_TARGET - context.sampleCount} more sample(s), or continue with lower reliability for demo purposes.`;
    }
    return 'Enrollment is ready. Submit the exam-day selfie to request exam access.';
}

function modelMismatchMessage(context) {
    return `This exam uses ${modelLabel(context.exam?.model_type)}; re-enroll for this model.`;
}

async function verifyStudent() {
    const examId = currentExamId();
    const studentId = document.getElementById('student-select').value;
    const file = document.getElementById('verify-file').files[0];
    if (!requireConsent()) return;
    if (!examId || !studentId || !file) {
        toast('Select an exam, student, and exam-day selfie.', 'error');
        return;
    }
    try {
        const result = await api.verifyStudent(examId, studentId, file);
        renderStudentResult(result);
        document.getElementById('verify-file').value = '';
        await refreshAll();
    } catch (err) {
        if (err.message.includes('enrolled with') || err.message.includes('requires')) {
            document.getElementById('enroll-warning').textContent = err.message;
            renderEnrollmentGuidance();
        }
        toast(err.message, 'error');
    }
}

async function verifyPreloadedStudent(fromSimulation = false) {
    const examId = currentExamId();
    const studentId = fromSimulation
        ? document.getElementById('simulation-student').value
        : document.getElementById('student-select').value;
    const consentId = fromSimulation ? 'simulation-consent-check' : 'consent-check';
    if (!document.getElementById(consentId).checked) {
        toast('Consent is required before using the preloaded selfie.', 'error');
        return;
    }
    if (!examId || !studentId) {
        toast('Select an exam and student first.', 'error');
        return;
    }
    try {
        if (fromSimulation) {
            setSelectValue('student-select', studentId);
        }
        const result = await api.verifyPreloadedStudent(examId, studentId, 'matching');
        state.simulationSelectedAttemptId = result.attempt.attempt_id;
        state.selectedAttempt = result.attempt;
        renderStudentResult(result);
        await refreshAll();
        await selectAttempt(result.attempt.attempt_id, { scroll: false });
    } catch (err) {
        if (err.message.includes('preuploaded') || err.message.includes('enrolled with')) {
            document.getElementById('enroll-warning').textContent = err.message;
            renderEnrollmentGuidance();
        }
        toast(err.message, 'error');
        renderSimulation();
    }
}

function renderStudentResult(result) {
    const attempt = result.attempt;
    state.studentResultAttemptId = attempt.attempt_id;
    const panel = document.getElementById('student-result');
    panel.classList.remove('verified', 'review', 'rejected');
    panel.classList.add(resultClass(attempt.decision));
    document.getElementById('result-title').textContent = result.message;
    document.getElementById('result-message').textContent = attempt.decision === 'verified'
        ? 'Exam access can be granted by the LMS.'
        : 'Use proctor review or manual ID fallback before granting access.';
    document.getElementById('result-score').textContent = attempt.score.toFixed(3);
    document.getElementById('result-threshold').textContent = attempt.threshold.toFixed(3);
    document.getElementById('result-attempt-id').textContent = attempt.attempt_id;
    document.getElementById('result-time').textContent = attempt.timestamp;
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
        toast(err.message, 'error');
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
        toast(err.message, 'error');
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
            `Imported ${result.imported.length}; rejected ${result.rejected.length}.`;
        toast('Roster import complete.', 'success');
        await refreshAll();
    } catch (err) {
        toast(err.message, 'error');
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
        ? `${eligible} eligible identities found | ${preuploaded} students preuploaded`
        : (status?.error || 'FLUXSynID dataset not found on this machine.');
    document.getElementById('flux-status').classList.toggle('warning', !available);
    document.getElementById('flux-status').classList.toggle('ready', available && preuploaded > 0);
    document.getElementById('flux-normalized-path').textContent = status?.normalized_path || '-';
    document.getElementById('flux-test-output').textContent = testSet?.output_dir || '-';
    document.getElementById('flux-test-count').textContent =
        `${exported} matching selfie${exported === 1 ? '' : 'ies'} exported`;

    const downloadLink = document.getElementById('flux-download-zip');
    downloadLink.href = api.fluxTestSetZipUrl();
    downloadLink.classList.toggle('hidden', exported <= 0);
}

async function preuploadFlux() {
    const button = document.getElementById('flux-preupload-btn');
    const priorText = button.textContent;
    button.disabled = true;
    button.textContent = 'Preuploading...';
    try {
        const result = await api.preuploadFlux({
            dataset_dir: valueOf('flux-dataset-dir'),
            count: valueOf('flux-count') || 25,
            seed: valueOf('flux-seed') || 42,
            model_type: valueOf('flux-model') || 'hybrid'
        });
        state.fluxStatus = result.status;
        state.fluxTestSet = result.export;
        renderFluxPanel(result.status, result.export);
        document.getElementById('flux-result').textContent =
            `Preuploaded ${result.imported_count} student(s); exported ${result.export?.image_count || 0} test selfie file(s); skipped ${result.skipped.length}.`;
        toast('FLUXSynID preupload complete.', 'success');
        await refreshAll();
    } catch (err) {
        document.getElementById('flux-result').textContent = err.message;
        toast(err.message, 'error');
    } finally {
        button.disabled = false;
        button.textContent = priorText;
    }
}

async function exportFluxTestSet() {
    const button = document.getElementById('flux-export-btn');
    const priorText = button.textContent;
    button.disabled = true;
    button.textContent = 'Exporting...';
    try {
        const result = await api.exportFluxTestSet();
        state.fluxTestSet = result;
        renderFluxPanel(state.fluxStatus, result);
        document.getElementById('flux-result').textContent =
            `Exported ${result.image_count} matching selfie file(s); skipped ${result.skipped.length}.`;
        toast('FLUXSynID test selfies exported.', 'success');
        renderGateContextPanel();
    } catch (err) {
        document.getElementById('flux-result').textContent = err.message;
        toast(err.message, 'error');
    } finally {
        button.disabled = false;
        button.textContent = priorText;
    }
}

async function resetDemo() {
    if (!window.confirm('Reset FaceVerify Campus demo data?')) return;
    try {
        const result = await api.resetDemo();
        resetReviewPanel();
        resetStudentResultPanel();
        const flux = result.flux_preupload;
        if (flux?.imported_count) {
            toast(`Demo reset with ${flux.imported_count} FLUXSynID faces.`, 'success');
        } else if (flux?.error) {
            toast(`Demo reset. FLUXSynID preupload skipped: ${flux.error}`, 'error');
        } else {
            toast('Demo reset.', 'success');
        }
        await refreshAll();
    } catch (err) {
        toast(err.message, 'error');
    }
}

function resetStudentResultPanel() {
    state.studentResultAttemptId = null;
    const panel = document.getElementById('student-result');
    panel.classList.remove('verified', 'review', 'rejected', 'active');
    document.getElementById('result-title').textContent = 'No attempt yet';
    document.getElementById('result-message').textContent = 'Select a student, consent, and submit a selfie.';
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
        toast(err.message, 'error');
    }
}

function renderAudit(rows) {
    const body = document.getElementById('audit-table');
    const latest = [...rows].reverse().slice(0, 12);
    body.innerHTML = latest.map(row => `
        <tr>
            <td>${escapeHtml(row.timestamp)}</td>
            <td>${escapeHtml(row.event_type)}</td>
            <td>${escapeHtml(row.actor)}</td>
            <td>${escapeHtml(row.message)}</td>
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
    state.simulationScenario = scenario;
    const targetStudentId = findSimulationScenarioStudent(scenario);
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
        state.simulationSelectedAttemptId = null;
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
    setTextIfPresent('simulation-student-guidance', simulationStudentGuidance(context, attempt));
    setPreview('simulation-student-camera-preview', attempt?.query_preview);

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
    setTextIfPresent('simulation-admin-time', attempt?.timestamp || '-');
    setPreview('simulation-reference-preview', row?.reference_preview || student?.reference_preview);
    setPreview('simulation-query-preview', attempt?.query_preview);
    renderAuditRail('simulation-audit-rail', attempt, row);

    document.querySelectorAll('[data-simulation-review-action]').forEach(button => {
        button.disabled = !attempt;
    });
}

function renderSimulationFeed() {
    const body = document.getElementById('simulation-feed-table');
    if (!body) return;
    const exam = selectedExam();
    const rows = [...(state.roster?.roster || [])]
        .sort((a, b) => statusPriority(a.exam_status) - statusPriority(b.exam_status)
            || String(a.name || '').localeCompare(String(b.name || '')))
        .slice(0, 5);

    if (!rows.length) {
        body.innerHTML = '<tr><td colspan="5" class="empty-cell">No students in this exam roster.</td></tr>';
        return;
    }

    body.innerHTML = rows.map(row => {
        const attempt = row.latest_attempt;
        const score = attempt ? `${(attempt.score * 100).toFixed(1)}%` : '-';
        const actionLabel = REVIEW_STATUSES.has(row.exam_status) || BLOCKED_STATUSES.has(row.exam_status) ? 'Review' : 'View';
        const action = attempt
            ? `<button class="row-button ${REVIEW_STATUSES.has(row.exam_status) || BLOCKED_STATUSES.has(row.exam_status) ? 'review-action' : ''}" data-simulation-attempt="${escapeHtml(attempt.attempt_id)}" data-simulation-student="${escapeHtml(row.student_id)}">${actionLabel}</button>`
            : '<span class="muted action-waiting">Waiting</span>';
        return `
            <tr class="${statusClass(row.exam_status)}">
                <td>${studentNameCell(row)}</td>
                <td>${escapeHtml(displayExamName(exam?.name))}</td>
                <td><span class="badge ${statusClass(row.exam_status)}">${escapeHtml(row.exam_status)}</span></td>
                <td>${score}</td>
                <td>${action}</td>
            </tr>
        `;
    }).join('');
}

async function openSimulationAttempt(attemptId, studentId) {
    if (studentId) {
        setSelectValue('student-select', studentId);
        setSelectValue('simulation-student', studentId);
        syncSimulationSelection();
    }
    state.simulationSelectedAttemptId = attemptId;
    await selectAttempt(attemptId, { scroll: false });
    renderSimulation();
}

function renderSimulationTimeline() {
    const list = document.getElementById('simulation-timeline-list');
    const rows = [...(state.auditRows || [])].reverse().slice(0, 8);
    if (!rows.length) {
        list.innerHTML = '<li class="empty-timeline">No campus events yet.</li>';
        return;
    }
    list.innerHTML = rows.map(row => `
        <li>
            <span>${escapeHtml(row.timestamp || '-')}</span>
            <strong>${escapeHtml(row.event_type || 'event')}</strong>
            <p>${escapeHtml(row.message || '-')}</p>
        </li>
    `).join('');
}

function renderAuditRail(containerId, attempt, row = null) {
    const host = document.getElementById(containerId);
    if (!host) return;
    if (!attempt) {
        host.innerHTML = '<div class="audit-empty-state">No verification attempt selected yet.</div>';
        return;
    }

    const stages = auditRailStages(attempt, row);
    host.innerHTML = stages.map((stage, index) => `
        <div class="audit-stage ${stage.kind}">
            <span class="audit-stage-icon">${escapeHtml(stage.icon)}</span>
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
    const livenessDetail = warnings.length ? 'Warnings present' : 'Passed';
    return [
        { icon: 'P', title: 'Attempt Started', detail: displayStudentId(attempt.student_id), time, kind: 'started' },
        { icon: 'C', title: 'Photo Captured', detail: attempt.query_preview ? 'Webcam / upload' : 'No photo preview', time, kind: 'captured' },
        { icon: 'L', title: 'Liveness Check', detail: livenessDetail, time, kind: warnings.length ? 'review' : 'passed' },
        { icon: 'F', title: 'Face Analyzed', detail: modelLabel(attempt.model_type), time, kind: 'analyzed' },
        { icon: '%', title: `Match Score: ${score}`, detail: scoreDetail(attempt), time, kind: resultClass(attempt.decision) },
        { icon: finalStatusIcon(finalStatus), title: finalStatus, detail: finalStatusDetail(finalStatus), time, kind: resultClass(attempt.decision) },
    ];
}

function scoreDetail(attempt) {
    const score = Number(attempt.score);
    const threshold = Number(attempt.threshold);
    if (!Number.isFinite(score) || !Number.isFinite(threshold)) return 'Awaiting threshold';
    if (score >= threshold) return 'At or above threshold';
    return 'Below threshold';
}

function finalStatusIcon(status) {
    if (ACCESS_GRANTED_STATUSES.has(status)) return 'G';
    if (REVIEW_STATUSES.has(status)) return 'R';
    if (BLOCKED_STATUSES.has(status)) return 'D';
    return 'W';
}

function finalStatusDetail(status) {
    if (ACCESS_GRANTED_STATUSES.has(status)) return 'Access can proceed';
    if (REVIEW_STATUSES.has(status)) return 'Auto-flagged';
    if (BLOCKED_STATUSES.has(status)) return 'Access denied';
    return 'Awaiting action';
}

function formatAuditTime(timestamp) {
    if (!timestamp) return '-';
    const parsed = new Date(timestamp);
    if (Number.isNaN(parsed.getTime())) return String(timestamp).split('T').pop() || timestamp;
    return parsed.toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
}

function renderSimulationPaneSwitch() {
    document.querySelectorAll('[data-simulation-pane]').forEach(button => {
        button.classList.toggle('active', button.dataset.simulationPane === state.simulationMobilePane);
    });
    document.querySelector('.student-pov')?.classList.toggle('mobile-active', state.simulationMobilePane === 'student');
    document.querySelector('.instructor-pov')?.classList.toggle('mobile-active', state.simulationMobilePane === 'instructor');
}

function simulationAttempt() {
    const row = currentRosterRow();
    if (state.selectedAttempt?.attempt_id === state.simulationSelectedAttemptId) return state.selectedAttempt;
    if (row?.latest_attempt?.attempt_id === state.simulationSelectedAttemptId) return row.latest_attempt;
    return row?.latest_attempt || null;
}

function simulationStudentGuidance(context, attempt) {
    if (!context.student) return 'Select a student to start the split-screen simulation.';
    if (context.modelMismatch) return modelMismatchMessage(context);
    if (context.sampleCount <= 0) return 'Enrollment is missing. The student cannot verify until face samples are enrolled.';
    if (context.sampleCount < ENROLLMENT_TARGET) return enrollmentGuidanceMessage(context);
    if (!document.getElementById('simulation-consent-check').checked) return 'Consent is required before the exam-day selfie can be submitted.';
    if (!attempt) return 'Enrollment is ready. Submit an exam-day selfie to see the instructor dashboard update.';
    return statusConsequence(attempt.final_status || attempt.status);
}

function studentFacingTitle(attempt) {
    if (ACCESS_GRANTED_STATUSES.has(attempt.final_status)) return 'Access granted';
    if (REVIEW_STATUSES.has(attempt.final_status)) return 'Manual review required';
    if (BLOCKED_STATUSES.has(attempt.final_status)) return 'Access blocked';
    return humanizeDecision(attempt.decision);
}

function studentFacingMessage(attempt) {
    if (ACCESS_GRANTED_STATUSES.has(attempt.final_status)) return 'You may continue into the exam.';
    if (attempt.final_status === 'Fallback Requested') return 'A proctor has requested a fallback ID check before exam access.';
    if (attempt.decision === 'manual_review') return 'A proctor must review this attempt before access is granted.';
    return 'This attempt did not meet the exam access policy.';
}

function setSimulationMiniStep(id, done, active) {
    const item = document.getElementById(id);
    item.classList.toggle('done', done);
    item.classList.toggle('active', active);
}

function setSimulationBadge(id, text, extraClass = '') {
    const badge = document.getElementById(id);
    badge.className = `badge ${statusClass(text)} ${extraClass}`.trim();
    badge.textContent = text || '-';
}

async function verifySimulationStudent() {
    const examId = currentExamId();
    const studentId = valueOf('simulation-student');
    const file = document.getElementById('simulation-verify-file').files[0];
    if (!document.getElementById('simulation-consent-check').checked) {
        toast('Consent is required before camera or upload verification.', 'error');
        return;
    }
    if (!examId || !studentId || !file) {
        toast('Select an exam, student, and exam-day selfie.', 'error');
        return;
    }
    try {
        setSelectValue('student-select', studentId);
        const result = await api.verifyStudent(examId, studentId, file);
        state.simulationSelectedAttemptId = result.attempt.attempt_id;
        state.selectedAttempt = result.attempt;
        renderStudentResult(result);
        document.getElementById('simulation-verify-file').value = '';
        await refreshAll();
        await selectAttempt(result.attempt.attempt_id, { scroll: false });
    } catch (err) {
        setTextIfPresent('simulation-student-guidance', err.message);
        toast(err.message, 'error');
        renderSimulation();
    }
}

async function reviewSimulationAttempt(action) {
    const attempt = simulationAttempt();
    if (!attempt) {
        toast('Submit or select an attempt before reviewing.', 'error');
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
        await selectAttempt(attempt.attempt_id, { scroll: false });
    } catch (err) {
        toast(err.message, 'error');
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
        document.getElementById('lab-title').textContent = result.match ? 'Likely Match' : 'Likely Different';
        document.getElementById('lab-score').textContent = result.score.toFixed(3);
        document.getElementById('lab-threshold').textContent = result.threshold.toFixed(3);
        document.getElementById('lab-decision').textContent = result.match ? 'Pass' : 'Review';
        updateScoreBar('lab', result.score, result.threshold, result.match ? 'verified' : 'rejected');
    } catch (err) {
        toast(err.message, 'error');
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
    setTextIfPresent(stepId, `Step ${Math.max(1, Math.min(4, stepNumber || 1))} of 4`);
    const pill = document.getElementById(pillId);
    if (!pill) return;
    pill.textContent = status?.label || 'Waiting';
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
    const contextSummary = [
        DISPLAY_UNIVERSITY,
        scenarioName || 'SE 204 Midterm',
        modelLabel(exam?.model_type || '-'),
        `Threshold ${thresholdText}`
    ].join(' · ');
    setTextIfPresent('context-summary', contextSummary);
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
    const title = `This exam requires ${model} verification before access.`;
    const detail = `Threshold ${thresholdText}. Higher score means stronger match. Students can request fallback ID review if not verified.`;
    setTextIfPresent('student-policy-title', title);
    setTextIfPresent('student-policy-detail', detail);
    setTextIfPresent('student-policy-model', model);
    setTextIfPresent('gate-policy-title', title);
    setTextIfPresent('gate-policy-detail', detail);
    setTextIfPresent('gate-policy-model', model);
    setTextIfPresent('admin-policy-title', title);
    setTextIfPresent('admin-policy-detail', `${detail} Model-specific enrollment is required.`);
    setTextIfPresent('admin-policy-model', model);
}

function renderGateContextPanel() {
    if (!document.getElementById('gate-selected-student')) return;
    const context = enrollmentContext();
    const row = context.rosterRow;
    const attempt = row?.latest_attempt;
    const studentText = context.student
        ? `${context.student.name} - ${displayStudentId(context.student.student_id)}`
        : 'Select a student';

    let enrollmentText = '-';
    if (!context.student) {
        enrollmentText = 'Select a student';
    } else if (context.modelMismatch) {
        enrollmentText = modelMismatchMessage(context);
    } else if (context.sampleCount <= 0) {
        enrollmentText = 'Enrollment needed before verification';
    } else if (context.sampleCount < ENROLLMENT_TARGET) {
        enrollmentText = `${context.sampleCount}/${ENROLLMENT_TARGET} samples; usable, but not ideal`;
    } else {
        enrollmentText = `Ready with ${context.sampleCount} sample(s)`;
    }

    const lastAttemptText = attempt
        ? `${attempt.final_status || row?.exam_status || 'Attempt recorded'} | score ${numberOrDash(attempt.score)} | ${attempt.timestamp || '-'}`
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

function updateScenarioRail(viewName) {
    const activeStep = viewName === 'launch' ? 'admin' : (viewName === 'lab' ? 'evidence' : viewName);
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
    return String(name).replace('CS 204', 'SE 204');
}

function displayExamName(name) {
    if (!name || name === 'Midterm 1') return DISPLAY_EXAM;
    return String(name);
}

function displayInstructor(name) {
    if (!name || name === 'Dr. Elena Morris') return 'Dr. Selin Demir';
    return String(name);
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
    return `${exam.start_time.slice(0, 16).replace('T', ' ')} - ${exam.end_time.slice(11, 16)}`;
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
        hybrid: 'Hybrid FaceNet'
    };
    return labels[modelType] || String(modelType || '-');
}

function humanizeDecision(decision) {
    const labels = {
        verified: 'Verified',
        manual_review: 'Manual Review',
        rejected: 'Rejected'
    };
    return labels[decision] || String(decision || '-');
}

function statusConsequence(status) {
    if (ACCESS_GRANTED_STATUSES.has(status)) return 'Access granted';
    if (REVIEW_STATUSES.has(status)) return 'Proctor action needed';
    if (BLOCKED_STATUSES.has(status)) return 'Exam access blocked';
    if (status === 'Enrollment Needed') return 'Enroll before verification';
    if (status === 'Not Started') return 'Waiting for student';
    if (status === 'Pending Verification') return 'Verification in progress';
    return 'Waiting';
}

function numberOrDash(value) {
    return Number.isFinite(Number(value)) ? Number(value).toFixed(3) : '-';
}

function setTextIfPresent(id, text) {
    const element = document.getElementById(id);
    if (element) element.textContent = text;
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
    item.textContent = message;
    host.appendChild(item);
    setTimeout(() => item.remove(), 4200);
}
