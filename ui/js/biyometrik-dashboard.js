/**
 * Biyometrik Sınav Erişim Doğrulaması — live controller.
 * Connects the standalone glassmorphic dashboard to the FaceVerify Campus
 * FastAPI backend (`api` from api.js) and the Turkish i18n helpers
 * (`window.statusLabel`, `decisionLabel`, `formatTime`, `formatDateTime`,
 * `apiErrorMessage` from i18n.js).
 *
 * Auto-picks the active exam, polls /campus every 5s (paused when the tab is
 * hidden), wires the Onayla / Reddet / Daha Sonra buttons to api.reviewAttempt,
 * and re-renders the activity chart from real attempt history.
 */
(() => {
    const POLL_MS = 5000;
    const REVIEWER = 'Dr. Selin Demir';
    const TR_DAYS = ['Pazar', 'Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi'];
    const TR_MONTHS = ['Ocak', 'Şubat', 'Mart', 'Nisan', 'Mayıs', 'Haziran',
                       'Temmuz', 'Ağustos', 'Eylül', 'Ekim', 'Kasım', 'Aralık'];

    const QUEUE_PAGE_SIZE = 5;

    // Decision constants (mirror inference/campus_store.py)
    const DEC_VERIFIED = 'verified';
    const DEC_REVIEW = 'manual_review';
    const DEC_REJECTED = 'rejected';

    // Final-status constants (terminal review outcomes)
    const FS_APPROVED = 'Approved by Proctor';
    const FS_REJECTED = 'Rejected';
    const FS_FALLBACK = 'Fallback Requested';
    const FS_REVIEW = 'Manual Review';

    const state = {
        examId: null,
        examIdLocked: false,
        snapshot: null,
        selectedAttemptId: null,
        activeFilter: 'review',
        page: 1,
        pollHandle: null,
        clockHandle: null,
        busy: false,
        lastErrorAt: 0,
        // diff-detection for live-update flash + sound
        firstRenderDone: false,
        lastRendered: {},          // keyed by element id → last rendered text
        lastQueueIds: new Map(),   // Map<attempt_id, status> visible in last queue render
        lastReviewIds: null,       // null on boot; Set on subsequent polls
        // queue UX (Feature 6)
        sortMode: 'priority',
        searchTerm: '',
        // Focus refactor — single source of truth for "who am I looking at"
        focusedStudentId: null,
        focusCleared: false,       // true while user explicitly closed (×); suppresses autoSelect
    };

    const LS_SORT_KEY = 'bx-sort-mode';

    // IDs that should show skeleton shimmer until first paint completes
    const SKELETON_TEXT_IDS = [
        'bx-kpi-total', 'bx-kpi-verified', 'bx-kpi-verified-pct',
        'bx-kpi-review', 'bx-kpi-review-pct',
        'bx-kpi-flagged', 'bx-kpi-flagged-pct',
        'bx-kpi-noreg', 'bx-kpi-noreg-pct',
        'bx-detail-name', 'bx-detail-id', 'bx-detail-exam',
        'bx-detail-score', 'bx-detail-time', 'bx-detail-attempt-id',
        'bx-gauge-value',
    ];

    // ── helpers ──────────────────────────────────────────────────────────
    const $ = (id) => document.getElementById(id);
    const pad = (n) => String(n).padStart(2, '0');

    function fmtClock(d) {
        let h = d.getHours();
        const m = pad(d.getMinutes());
        const ampm = h >= 12 ? 'PM' : 'AM';
        h = h % 12 || 12;
        return `${h}:${m} ${ampm}`;
    }
    function fmtClockSec(d) {
        let h = d.getHours();
        const m = pad(d.getMinutes());
        const s = pad(d.getSeconds());
        const ampm = h >= 12 ? 'PM' : 'AM';
        h = h % 12 || 12;
        return `${h}:${m}:${s} ${ampm}`;
    }
    function fmtDate(d) {
        return `${d.getDate()} ${TR_MONTHS[d.getMonth()]} ${d.getFullYear()}, ${TR_DAYS[d.getDay()]}`;
    }
    function fmtRowTime(iso) {
        if (!iso) return '—';
        const d = new Date(iso);
        if (Number.isNaN(d.getTime())) return '—';
        return fmtClock(d);
    }
    function fmtFullTime(iso) {
        if (!iso) return '—';
        const d = new Date(iso);
        if (Number.isNaN(d.getTime())) return '—';
        return fmtClockSec(d);
    }
    function fmtShortDate(iso) {
        if (!iso) return '—';
        const d = new Date(iso);
        if (Number.isNaN(d.getTime())) return '—';
        return `${d.getDate()} ${TR_MONTHS[d.getMonth()]} ${d.getFullYear()}`;
    }
    function pct(num, denom) {
        if (!denom) return '%0';
        return `%${(num / denom * 100).toFixed(1)}`;
    }

    function safeText(el, value) {
        if (el && el.textContent !== String(value)) el.textContent = String(value);
    }

    // ── Focus refactor: central setters ──────────────────────────────────
    // Returns true if `el` is the currently-focused element (so polling
    // refresh skips writes that would clobber an in-flight user interaction).
    function isUserEditing(el) {
        return !!el && document.activeElement === el;
    }

    // Single entry point for changing "who the user is looking at".
    // Derives missing field, keeps simState.studentId in lockstep,
    // and re-renders queue + detail + simulation toolbar atomically.
    function setFocus({ studentId = null, attemptId = null, source = 'user' } = {}) {
        // Derive the missing field from the snapshot
        if (attemptId && !studentId) {
            const att = findAttempt(state.snapshot, attemptId);
            studentId = att?.student_id || null;
        }
        if (studentId && !attemptId && state.snapshot && state.examId) {
            // Most-recent attempt of this student in current exam (attempts are
            // returned timestamp-desc; first match is freshest)
            const att = (state.snapshot.attempts || []).find(
                a => a.exam_id === state.examId && a.student_id === studentId
            );
            attemptId = att?.attempt_id || null;
        }

        state.focusedStudentId = studentId;
        state.selectedAttemptId = attemptId;
        if (typeof simState !== 'undefined') {
            simState.studentId = studentId;
            simState.userPickedStudent = (source === 'user-toolbar');
        }
        state.focusCleared = (studentId === null && attemptId === null && source === 'user-close');

        if (state.snapshot && state.examId) {
            renderQueue(state.snapshot, state.examId);
            renderSelected(state.snapshot, state.examId);
            if (typeof renderSimulationToolbar === 'function') {
                renderSimulationToolbar(state.snapshot, state.examId);
            }
        }
    }

    function setFilter(filter) {
        state.activeFilter = filter;
        state.page = 1;
        document.querySelectorAll('#bx-queue-tabs .bx-tab').forEach(b =>
            b.classList.toggle('is-active', b.dataset.filter === filter));
        if (state.snapshot && state.examId) {
            autoSelectIfNeeded(state.snapshot, state.examId);
            renderQueue(state.snapshot, state.examId);
            renderSelected(state.snapshot, state.examId);
        }
    }

    function setExam(examId, { locked = true } = {}) {
        state.examId = examId;
        state.examIdLocked = locked;
        try { persistExamSelection(locked ? examId : 'auto'); } catch (e) {}
        // Reset everything that depends on exam
        state.selectedAttemptId = null;
        state.focusedStudentId = null;
        state.focusCleared = false;
        state.activeFilter = 'review';
        state.searchTerm = '';
        state.page = 1;
        state.lastQueueIds = new Map();
        if (typeof simState !== 'undefined') {
            simState.studentId = null;
            simState.userPickedStudent = false;
            simState.stagedFile = null;
            simState.consent = false;
        }
        // Sync DOM controls so the user sees a clean reset
        const searchInput = $('bx-queue-search');
        if (searchInput) searchInput.value = '';
        const fileInput = $('bx-sim-file');
        if (fileInput) fileInput.value = '';
        document.querySelectorAll('#bx-queue-tabs .bx-tab').forEach(b =>
            b.classList.toggle('is-active', b.dataset.filter === 'review'));
        refresh();
    }

    // ── skeleton loader (Feature 1) ──────────────────────────────────────
    function applySkeletons(on) {
        for (const id of SKELETON_TEXT_IDS) {
            const el = $(id);
            if (el) el.classList.toggle('bx-skeleton', !!on);
        }
        const tbody = $('bx-queue-tbody');
        if (tbody && on) {
            // Render 4 placeholder rows
            tbody.innerHTML = Array.from({ length: 4 }).map(() => `
                <tr class="bx-skeleton-row">
                    <td><span class="bx-sk-block" style="width:140px;height:36px;border-radius:18px"></span></td>
                    <td><span class="bx-sk-block" style="width:64px"></span></td>
                    <td><span class="bx-sk-block" style="width:40px"></span></td>
                    <td><span class="bx-sk-block" style="width:64px;height:18px;border-radius:9px"></span></td>
                    <td><span class="bx-sk-block" style="width:56px"></span></td>
                    <td><span class="bx-sk-block" style="width:60px;height:24px;border-radius:6px"></span></td>
                </tr>
            `).join('');
        }
    }

    // ── live-update flash (Feature 2) ────────────────────────────────────
    function flashChanged(el, newValue, key) {
        if (!el) return;
        const v = String(newValue);
        if (state.firstRenderDone && state.lastRendered[key] !== undefined && state.lastRendered[key] !== v) {
            el.classList.remove('bx-flash');
            // force reflow so the animation can replay
            void el.offsetWidth;
            el.classList.add('bx-flash');
        }
        state.lastRendered[key] = v;
    }
    function safeAttr(el, attr, value) {
        if (el && el.getAttribute(attr) !== String(value)) el.setAttribute(attr, value);
    }

    // ── toast ────────────────────────────────────────────────────────────
    let toastTimer = null;
    function toast(msg, kind = '') {
        const el = $('bx-toast');
        if (!el) return;
        el.textContent = msg;
        el.classList.remove('is-success', 'is-error');
        if (kind) el.classList.add(`is-${kind}`);
        el.classList.add('is-visible');
        el.setAttribute('aria-hidden', 'false');
        if (toastTimer) clearTimeout(toastTimer);
        toastTimer = setTimeout(() => {
            el.classList.remove('is-visible');
            el.setAttribute('aria-hidden', 'true');
        }, 2800);
    }

    function reportError(label, err) {
        const now = Date.now();
        // Throttle error toasts so a dead backend doesn't spam the screen
        if (now - state.lastErrorAt < 4000) return;
        state.lastErrorAt = now;
        const translator = apiErrorMessage || ((e) => (e && e.message) || String(e));
        toast(`${label}: ${translator(err)}`, 'error');
        // eslint-disable-next-line no-console
        console.error(label, err);
    }

    // ── i18n shims (graceful fallback if i18n.js failed to load) ─────────
    const tStatus = (s) => (window.statusLabel ? window.statusLabel(s) : s || '—');
    const tDecision = (s) => (window.decisionLabel ? window.decisionLabel(s) : s || '—');

    // ── clock ────────────────────────────────────────────────────────────
    function tickClock() {
        const now = new Date();
        safeText($('bx-clock-time'), fmtClock(now));
        safeText($('bx-clock-date'), fmtDate(now));
        safeText($('bx-last-update'), fmtClockSec(now));
    }

    // ── exam selection ───────────────────────────────────────────────────
    function pickActiveExam(snapshot) {
        if (!snapshot) return null;
        const attempts = snapshot.attempts || [];
        // Snapshots are returned timestamp-DESC, so the first is the most recent.
        const latest = attempts[0];
        if (latest && latest.exam_id) return latest.exam_id;
        const exams = snapshot.exams || [];
        return exams.length ? exams[0].exam_id : null;
    }

    function findExam(snapshot, examId) {
        return (snapshot.exams || []).find(e => e.exam_id === examId) || null;
    }
    function findCourse(snapshot, courseId) {
        return (snapshot.courses || []).find(c => c.course_id === courseId) || null;
    }
    function findStudent(snapshot, studentId) {
        return (snapshot.students || []).find(s => s.student_id === studentId) || null;
    }
    function findAttempt(snapshot, attemptId) {
        return (snapshot.attempts || []).find(a => a.attempt_id === attemptId) || null;
    }

    // ── KPI / aggregation ────────────────────────────────────────────────
    function effectiveDecision(attempt) {
        // After a manual review, final_status overrides decision for filtering.
        const fs = attempt.final_status;
        if (fs === FS_APPROVED) return DEC_VERIFIED;
        if (fs === FS_REJECTED) return DEC_REJECTED;
        if (fs === FS_FALLBACK) return DEC_REVIEW;
        return attempt.decision;
    }

    function isOpenForReview(attempt) {
        // Rows that should show in the "İnceleme" tab and accept Approve/Deny.
        const fs = attempt.final_status;
        if (fs === FS_APPROVED || fs === FS_REJECTED) return false;
        return attempt.decision === DEC_REVIEW || fs === FS_FALLBACK;
    }

    function attemptsForExam(snapshot, examId) {
        return (snapshot.attempts || []).filter(a => a.exam_id === examId);
    }

    function rosterForExam(snapshot, exam) {
        if (!exam) return [];
        const courseId = exam.course_id;
        return (snapshot.students || []).filter(s =>
            (s.course_ids || []).includes(courseId)
        );
    }

    function computeKpis(snapshot, examId, exam) {
        const examAttempts = attemptsForExam(snapshot, examId);
        const total = examAttempts.length;
        let verified = 0, review = 0, rejected = 0;
        for (const a of examAttempts) {
            const d = effectiveDecision(a);
            if (d === DEC_VERIFIED) verified++;
            else if (d === DEC_REVIEW) review++;
            else if (d === DEC_REJECTED) rejected++;
        }
        const roster = rosterForExam(snapshot, exam);
        const noEnrollment = roster.filter(s => Number(s.sample_count || 0) === 0).length;
        return {
            total,
            verified,
            verifiedPct: pct(verified, total),
            review,
            reviewPct: pct(review, total),
            rejected,
            rejectedPct: pct(rejected, total),
            noEnrollment,
            noEnrollmentPct: pct(noEnrollment, roster.length),
        };
    }

    // ── render: topbar ───────────────────────────────────────────────────
    function renderTopbar(snapshot, examId) {
        const exam = findExam(snapshot, examId);
        if (!exam) return;
        const course = findCourse(snapshot, exam.course_id);
        const courseName = course ? course.name : exam.course_id;
        const examName = exam.name || '—';
        const model = exam.model_type || '—';
        const threshold = Number.isFinite(exam.threshold) ? exam.threshold.toFixed(3) : '—';
        const sub = $('bx-topbar-subtitle');
        if (sub) {
            sub.innerHTML = '';
            const mk = (txt, cls) => {
                const span = document.createElement('span');
                if (cls) span.className = cls;
                span.textContent = txt;
                return span;
            };
            sub.appendChild(document.createTextNode('Atılım University Demo '));
            sub.appendChild(mk('•', 'bx-sep'));
            sub.appendChild(document.createTextNode(` ${courseName} — ${examName} `));
            sub.appendChild(mk('•', 'bx-sep'));
            sub.appendChild(document.createTextNode(` ${model} `));
            sub.appendChild(mk('•', 'bx-sep'));
            sub.appendChild(document.createTextNode(` Eşik: ${threshold}`));
        }
    }

    // ── render: KPIs + sidebar badges + bell ─────────────────────────────
    function renderKpis(kpis) {
        const set = (id, val) => {
            const el = $(id);
            safeText(el, val);
            flashChanged(el, val, id);
        };
        set('bx-kpi-total', kpis.total);
        set('bx-kpi-verified', kpis.verified);
        set('bx-kpi-verified-pct', kpis.verifiedPct);
        set('bx-kpi-review', kpis.review);
        set('bx-kpi-review-pct', kpis.reviewPct);
        set('bx-kpi-flagged', kpis.rejected);
        set('bx-kpi-flagged-pct', kpis.rejectedPct);
        set('bx-kpi-noreg', kpis.noEnrollment);
        set('bx-kpi-noreg-pct', kpis.noEnrollmentPct);
        set('bx-nav-badge-review', kpis.review);
        set('bx-nav-badge-flagged', kpis.rejected);
        // Bell badge: only owned by renderKpis until audit feed loads.
        // After audit loads, updateAuditBadge takes over (unread-since-last-open).
        if (auditState.lastEntries.length === 0) {
            const bell = $('bx-bell-count');
            if (bell) {
                const v = Math.max(kpis.review, 0);
                const s = String(v);
                if (bell.textContent !== s) {
                    bell.textContent = s;
                    if (state.firstRenderDone) flashChanged(bell, v, 'bx-bell-count');
                    else state.lastRendered['bx-bell-count'] = s;
                }
                bell.style.display = v ? '' : 'none';
            }
        }
    }

    // ── queue rendering ──────────────────────────────────────────────────
    function passesFilter(attempt, snapshot, filter) {
        if (filter === 'all') return true;
        if (filter === 'review') return isOpenForReview(attempt);
        if (filter === 'flagged') return effectiveDecision(attempt) === DEC_REJECTED;
        if (filter === 'no_enrollment') {
            const student = findStudent(snapshot, attempt.student_id);
            return !!student && Number(student.sample_count || 0) === 0;
        }
        return true;
    }

    function tabCounts(snapshot, examId) {
        const list = attemptsForExam(snapshot, examId);
        const counts = { all: list.length, review: 0, flagged: 0, no_enrollment: 0 };
        for (const a of list) {
            if (isOpenForReview(a)) counts.review++;
            if (effectiveDecision(a) === DEC_REJECTED) counts.flagged++;
            const student = findStudent(snapshot, a.student_id);
            if (student && Number(student.sample_count || 0) === 0) counts.no_enrollment++;
        }
        return counts;
    }

    function priorityRank(attempt) {
        // Lower rank = higher priority (renders first)
        const d = effectiveDecision(attempt);
        if (d === DEC_REVIEW) return 0;
        if (d === DEC_REJECTED) return 1;
        if (d === DEC_VERIFIED) return 2;
        return 3;
    }

    function sortQueue(rows) {
        const mode = state.sortMode;
        const arr = rows.slice();
        if (mode === 'newest') {
            return arr.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
        }
        if (mode === 'oldest') {
            return arr.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
        }
        if (mode === 'score-asc') {
            return arr.sort((a, b) => (a.score || 0) - (b.score || 0));
        }
        if (mode === 'score-desc') {
            return arr.sort((a, b) => (b.score || 0) - (a.score || 0));
        }
        // default: priority
        return arr.sort((a, b) => {
            const pa = priorityRank(a), pb = priorityRank(b);
            if (pa !== pb) return pa - pb;
            if (a.score !== b.score) return a.score - b.score;
            return new Date(b.timestamp) - new Date(a.timestamp);
        });
    }

    function matchesSearch(attempt) {
        const q = state.searchTerm.trim().toLowerCase();
        if (!q) return true;
        const name = String(attempt.student_name || '').toLowerCase();
        const id = String(attempt.student_id || '').toLowerCase();
        return name.includes(q) || id.includes(q);
    }

    function scoreClass(attempt) {
        const score = Number(attempt.score) || 0;
        const threshold = Number(attempt.threshold) || 0;
        if (score >= threshold) return 'bx-score-green';
        if (score >= threshold * 0.85) return 'bx-score-orange';
        return 'bx-score-red';
    }

    function tagForDecision(attempt) {
        const d = effectiveDecision(attempt);
        if (d === DEC_REVIEW) return { label: 'İnceleme', cls: 'bx-tag-orange' };
        if (d === DEC_REJECTED) return { label: 'İşaretli', cls: 'bx-tag-red' };
        if (d === DEC_VERIFIED) {
            if (attempt.final_status === FS_APPROVED) return { label: 'Onaylandı', cls: 'bx-tag-green' };
            return { label: 'Doğrulandı', cls: 'bx-tag-green' };
        }
        return { label: tDecision(d), cls: 'bx-tag-orange' };
    }

    function fallbackImageForDecision(attempt) {
        // When query_preview is missing, pick a face from the bundled pool.
        const id = String(attempt.student_id || '');
        if (id.includes('1042') || /aylin/i.test(attempt.student_name || '')) return 'img/aylin_kaya_live.jpg';
        if (id.includes('1053')) return 'img/elif_bayar.jpg';
        if (id.includes('1061')) return 'img/sofia_martinez.jpg';
        if (id.includes('1045')) return 'img/nora_walker.jpg';
        if (id.includes('1066')) return 'img/isabella_rossi.jpg';
        return 'img/aylin_kaya_live.jpg';
    }

    function renderQueue(snapshot, examId) {
        const tbody = $('bx-queue-tbody');
        const tabs = $('bx-queue-tabs');
        if (!tbody) return;

        // Update tab counters
        if (tabs) {
            const counts = tabCounts(snapshot, examId);
            tabs.querySelectorAll('b[data-count]').forEach(b => {
                const key = b.getAttribute('data-count');
                b.textContent = `(${counts[key] ?? 0})`;
            });
        }

        const rows = sortQueue(
            attemptsForExam(snapshot, examId)
                .filter(a => passesFilter(a, snapshot, state.activeFilter))
                .filter(matchesSearch)
        );
        const total = rows.length;
        const startIdx = (state.page - 1) * QUEUE_PAGE_SIZE;
        const pageRows = rows.slice(startIdx, startIdx + QUEUE_PAGE_SIZE);

        // Empty state
        if (pageRows.length === 0) {
            tbody.innerHTML =
                '<tr><td colspan="6" class="bx-queue-empty">Bu kategoride deneme yok.</td></tr>';
            updatePagerAndFooter(0, 0, 0);
            return;
        }

        const exam = findExam(snapshot, examId);
        const examLabel = exam ? (exam.name || 'Sınav') : 'Sınav';

        // Map<attempt_id, status> — flash only when an attempt_id is genuinely
        // new (was not visible in the previous render). Status changes alone
        // (e.g. manual_review → approved) don't trigger a re-flash.
        const previousIds = state.lastQueueIds instanceof Map
            ? state.lastQueueIds
            : new Map();
        const nextIds = new Map(pageRows.map(r => [r.attempt_id, r.final_status || r.status || '']));

        tbody.innerHTML = '';
        for (const a of pageRows) {
            const tr = document.createElement('tr');
            tr.dataset.attemptId = a.attempt_id;
            if (a.attempt_id === state.selectedAttemptId) tr.classList.add('is-selected');
            // Flash only if this attempt_id wasn't in the previous render (genuinely new)
            if (state.firstRenderDone && !previousIds.has(a.attempt_id)) {
                tr.classList.add('bx-flash');
            }

            const tag = tagForDecision(a);
            const scorePct = Math.round((Number(a.score) || 0) * 100);
            const photoUrl = a.query_preview || fallbackImageForDecision(a);
            const studentName = a.student_name || '—';
            const studentId = a.student_id || '—';

            tr.innerHTML = `
                <td>
                    <div class="bx-queue-user">
                        <img src="${escapeAttr(photoUrl)}" alt="" loading="lazy">
                        <div><strong>${escapeHtml(studentName)}</strong><span>${escapeHtml(studentId)}</span></div>
                    </div>
                </td>
                <td class="bx-q-muted">${escapeHtml(examLabel)}</td>
                <td><span class="bx-score ${scoreClass(a)}">${scorePct}%</span></td>
                <td><span class="bx-tag ${tag.cls}">${escapeHtml(tag.label)}</span></td>
                <td class="bx-q-muted bx-q-mono">${escapeHtml(fmtRowTime(a.timestamp))}</td>
                <td>
                    <button class="bx-btn-ghost-orange" data-action="inspect">İncele</button>
                </td>
            `;
            tbody.appendChild(tr);
        }

        updatePagerAndFooter(total, startIdx + 1, startIdx + pageRows.length);
        state.lastQueueIds = nextIds;
    }

    function updatePagerAndFooter(total, fromIdx, toIdx) {
        const info = $('bx-queue-foot-info');
        if (info) {
            info.textContent = total
                ? `${total} kayıttan ${fromIdx} - ${toIdx} arası gösteriliyor`
                : 'Kayıt yok';
        }
        const pager = $('bx-queue-pager');
        if (!pager) return;
        const pages = Math.max(1, Math.ceil(total / QUEUE_PAGE_SIZE));
        pager.innerHTML = '';
        const maxButtons = Math.min(pages, 4);
        for (let i = 1; i <= maxButtons; i++) {
            const b = document.createElement('button');
            b.dataset.page = String(i);
            b.textContent = String(i);
            if (i === state.page) b.classList.add('is-active');
            pager.appendChild(b);
        }
        if (pages > maxButtons) {
            const ellipsis = document.createElement('span');
            ellipsis.textContent = '…';
            pager.appendChild(ellipsis);
        }
        if (pages > 1) {
            const next = document.createElement('button');
            next.className = 'bx-pager-arrow';
            next.setAttribute('aria-label', 'Sonraki');
            next.dataset.page = String(Math.min(pages, state.page + 1));
            next.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m9 6 6 6-6 6"/></svg>';
            pager.appendChild(next);
        }
    }

    function escapeHtml(str) {
        return String(str ?? '').replace(/[&<>"']/g, (c) => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
        }[c]));
    }
    function escapeAttr(str) { return escapeHtml(str); }

    // ── selected attempt detail ──────────────────────────────────────────
    function autoSelectIfNeeded(snapshot, examId) {
        // User explicitly closed the detail panel — respect their intent until
        // the next deliberate action (queue click, chip click, etc.)
        if (state.focusCleared) return;

        // If the current selection is still valid in this view, keep it
        if (state.selectedAttemptId) {
            const existing = findAttempt(snapshot, state.selectedAttemptId);
            if (existing && passesFilter(existing, snapshot, state.activeFilter) && matchesSearch(existing)) {
                state.focusedStudentId = existing.student_id;
                return;
            }
        }
        // Try to preserve focusedStudentId by finding their most-recent
        // attempt that passes the current view filters
        if (state.focusedStudentId) {
            const att = (snapshot.attempts || []).find(
                a => a.exam_id === examId
                  && a.student_id === state.focusedStudentId
                  && passesFilter(a, snapshot, state.activeFilter)
                  && matchesSearch(a)
            );
            if (att) {
                state.selectedAttemptId = att.attempt_id;
                return;
            }
        }
        // Fall back: first attempt in the current queue view
        const queue = sortQueue(
            attemptsForExam(snapshot, examId)
                .filter(a => passesFilter(a, snapshot, state.activeFilter))
                .filter(matchesSearch)
        );
        if (queue.length) {
            state.selectedAttemptId = queue[0].attempt_id;
            state.focusedStudentId = queue[0].student_id;
        } else {
            state.selectedAttemptId = null;
            state.focusedStudentId = null;
        }
    }

    function renderSelected(snapshot, examId) {
        const attempt = state.selectedAttemptId ? findAttempt(snapshot, state.selectedAttemptId) : null;
        const exam = findExam(snapshot, examId);
        const student = attempt ? findStudent(snapshot, attempt.student_id) : null;

        // Header attempt id
        safeText($('bx-detail-attempt-id'), attempt ? attempt.attempt_id : '—');

        // Top stats
        const heroImg = $('bx-hero-img');
        const scoreEl = $('bx-detail-score');
        const scoreLabelEl = $('bx-detail-score-label');
        const thresholdEl = $('bx-detail-threshold');
        const timeEl = $('bx-detail-time');
        const dateEl = $('bx-detail-date');
        const nameEl = $('bx-detail-name');
        const idEl = $('bx-detail-id');
        const examEl = $('bx-detail-exam');
        const avatarEl = $('bx-detail-avatar');

        if (!attempt) {
            // No attempt yet — leave hero placeholder, blank-out detail values
            safeText(nameEl, '—');
            safeText(idEl, '—');
            safeText(examEl, exam ? (exam.name || '—') : '—');
            safeText(scoreEl, '—');
            safeText(scoreLabelEl, '');
            safeText(thresholdEl, exam && Number.isFinite(exam.threshold) ? exam.threshold.toFixed(3) : '—');
            safeText(timeEl, '—');
            safeText(dateEl, '');
            const bar = $('bx-detail-confidence-bar');
            if (bar) bar.style.width = '0%';
            safeText($('bx-detail-confidence-label'), '—');
            renderEvidence(null, null);
            renderGauge(null);
            renderAiList(null, exam);
            renderTechMeta(null, exam);
            renderSteps(null, null);
            renderHeroCaption(null, null);
            hideReverifyBadge();
            return;
        }

        const photoUrl = attempt.query_preview || fallbackImageForDecision(attempt);
        if (heroImg) safeAttr(heroImg, 'src', photoUrl);
        if (avatarEl) safeAttr(avatarEl, 'src', photoUrl);

        safeText(nameEl, attempt.student_name || (student && student.name) || '—');
        safeText(idEl, attempt.student_id || '—');
        safeText(examEl, exam ? (exam.name || 'Sınav') : 'Sınav');

        const scorePct = Math.round((Number(attempt.score) || 0) * 100);
        safeText(scoreEl, `${scorePct}%`);
        scoreEl?.classList.remove('bx-text-green');
        if (scoreClass(attempt) === 'bx-score-green') scoreEl?.classList.add('bx-text-green');

        safeText(scoreLabelEl, scorePct >= 80 ? 'Yüksek Eşleşme' : scorePct >= 60 ? 'Orta Eşleşme' : 'Düşük Eşleşme');

        safeText(thresholdEl, Number.isFinite(attempt.threshold) ? Number(attempt.threshold).toFixed(3) : '—');
        safeText(timeEl, fmtFullTime(attempt.timestamp));
        safeText(dateEl, fmtShortDate(attempt.timestamp));

        const bar = $('bx-detail-confidence-bar');
        const label = $('bx-detail-confidence-label');
        if (bar && label) {
            const w = Math.min(100, Math.max(0, scorePct));
            bar.style.width = `${w}%`;
            const lbl = w >= 80 ? 'Yüksek' : w >= 60 ? 'Orta' : 'Düşük';
            label.textContent = lbl;
        }

        renderEvidence(attempt, student);
        renderGauge(attempt);
        renderAiList(attempt, exam);
        renderTechMeta(attempt, exam);
        renderSteps(attempt, student);
        renderHeroCaption(attempt, student);
        // Kick off diff-map computation asynchronously (no await; UI doesn't block)
        renderDiffMap(attempt, student);
        // Apply cached re-verify badge if user already ran it for this attempt
        applyCachedReverify(attempt.attempt_id);

        // Sync queue selection class
        document.querySelectorAll('#bx-queue-tbody tr').forEach(tr => {
            tr.classList.toggle('is-selected', tr.dataset.attemptId === attempt.attempt_id);
        });
    }

    function renderEvidence(attempt, student) {
        const liveImg = $('bx-evidence-live');
        const enrolledImg = $('bx-evidence-enrolled');
        const diffImg = $('bx-evidence-diff');
        const featureImg = $('bx-evidence-feature');
        const heroImg = $('bx-hero-img');

        // Reset the diff-computed state before each render — renderDiffMap will
        // re-apply it if computation succeeds for the new attempt.
        diffImg?.closest('.bx-evidence-img')?.classList.remove('bx-diff-computed');

        const queryUrl = attempt && attempt.query_preview
            ? attempt.query_preview
            : (attempt ? fallbackImageForDecision(attempt) : 'img/aylin_kaya_live.jpg');
        const refUrl = student && student.reference_preview
            ? student.reference_preview
            : (attempt ? fallbackImageForDecision(attempt) : 'img/aylin_kaya_enrolled.jpg');

        if (liveImg) safeAttr(liveImg, 'src', queryUrl);
        if (enrolledImg) safeAttr(enrolledImg, 'src', refUrl);
        if (diffImg) safeAttr(diffImg, 'src', queryUrl);
        if (featureImg) safeAttr(featureImg, 'src', refUrl);
        if (heroImg && attempt) safeAttr(heroImg, 'src', queryUrl);
    }

    function renderGauge(attempt) {
        const valEl = $('bx-gauge-value');
        const ringEl = $('bx-gauge-ring');
        const labelEl = $('bx-gauge-label');
        if (!attempt) {
            safeText(valEl, '—');
            safeText(labelEl, 'BEKLEMEDE');
            if (ringEl) ringEl.setAttribute('stroke-dashoffset', '326.7');
            return;
        }
        const scorePct = Math.round((Number(attempt.score) || 0) * 100);
        safeText(valEl, `${scorePct}%`);
        if (ringEl) {
            const circumference = 326.7;
            const offset = circumference * (1 - scorePct / 100);
            ringEl.setAttribute('stroke-dashoffset', String(Math.max(0, offset).toFixed(1)));
        }
        const label = scorePct >= 80 ? 'YÜKSEK EŞLEŞME'
                    : scorePct >= 60 ? 'ORTA EŞLEŞME'
                    : 'DÜŞÜK EŞLEŞME';
        safeText(labelEl, label);
    }

    function renderAiList(attempt, exam) {
        const ul = $('bx-ai-list');
        if (!ul) return;
        if (!attempt) return;  // leave whatever's there
        const bullets = deriveAiAnalysis(attempt, exam);
        ul.innerHTML = bullets.map(b => `<li>${escapeHtml(b)}</li>`).join('');
    }

    function deriveAiAnalysis(attempt, exam) {
        const bullets = [];
        const score = Number(attempt.score) || 0;
        const threshold = Number(attempt.threshold) || 0;
        const delta = score - threshold;

        // Score vs threshold (always include)
        if (delta > 0.10) {
            bullets.push(`Skor eşiğin %${(delta * 100).toFixed(1)} üzerinde — güçlü eşleşme`);
        } else if (delta > 0) {
            bullets.push(`Skor eşiğin sadece %${(delta * 100).toFixed(1)} üzerinde — sınır çizgisinde`);
        } else if (delta > -0.05) {
            bullets.push(`Skor eşiğin %${(-delta * 100).toFixed(1)} altında — manuel inceleme gerekli`);
        } else {
            bullets.push(`Skor eşiğin %${(-delta * 100).toFixed(1)} altında — düşük güven`);
        }

        // Validation confidence
        const validation = attempt.validation || {};
        if (typeof validation.confidence === 'number' && validation.confidence < 0.85) {
            bullets.push(`Görüntü kalite güveni: %${(validation.confidence * 100).toFixed(0)}`);
        } else if (typeof validation.confidence === 'number') {
            bullets.push(`Görüntü kalitesi iyi (güven: %${(validation.confidence * 100).toFixed(0)})`);
        }

        // Backend warnings (translated via i18n if possible)
        const warnings = Array.isArray(attempt.warnings) ? attempt.warnings : [];
        const translate = window.localizeText || ((s) => s);
        for (const w of warnings) {
            const translated = translate(String(w));
            bullets.push(translated);
        }

        // Source
        if (attempt.attempt_source === 'preloaded' || attempt.scenario) {
            bullets.push('Hazır demo selfiesi (sentetik FLUXSynID)');
        }

        // Review state
        if (attempt.review) {
            const action = attempt.review.action;
            const actionTr = action === 'approve' ? 'onay'
                          : action === 'deny' ? 'ret'
                          : action === 'fallback' ? 'yedek kontrol' : action;
            bullets.push(`İnceleme: ${actionTr} — ${attempt.review.reviewer || 'Gözetmen'}`);
        }

        // Time-of-attempt vs exam window
        if (exam && attempt.timestamp) {
            const t = new Date(attempt.timestamp).getTime();
            const start = exam.start_time ? new Date(exam.start_time).getTime() : NaN;
            const end = exam.end_time ? new Date(exam.end_time).getTime() : NaN;
            if (!Number.isNaN(start) && t < start) {
                bullets.push('Deneme sınav başlamadan önce gerçekleşti');
            } else if (!Number.isNaN(end) && t > end) {
                bullets.push('Deneme sınav penceresi dışında');
            }
        }

        // Cap at 6, pad to 4 with em-dash
        while (bullets.length < 4) bullets.push('—');
        return bullets.slice(0, 6);
    }

    function renderTechMeta(attempt, exam) {
        safeText($('bx-ai-meta-model'),
            (attempt && attempt.model_type) || (exam && exam.model_type) || '—');
        safeText($('bx-ai-meta-decision'),
            attempt
                ? tStatus(attempt.final_status || attempt.status)
                : '—');
        let source = '—';
        if (attempt) {
            if (attempt.attempt_source === 'preloaded' || attempt.scenario) source = 'Hazır demo selfiesi';
            else if (attempt.attempt_source === 'upload') source = 'Yüklenen selfie';
            else if (attempt.attempt_source) source = attempt.attempt_source;
        }
        safeText($('bx-ai-meta-source'), source);
    }

    // ── activity chart ───────────────────────────────────────────────────
    function renderChart(snapshot, examId) {
        const svg = $('bx-chart');
        if (!svg) return;
        const examAttempts = attemptsForExam(snapshot, examId);

        // Bucket by 10 min, last 60 min ⇒ 6 buckets
        const now = new Date();
        const bucketMs = 10 * 60 * 1000;
        const buckets = 6;
        const start = now.getTime() - buckets * bucketMs;

        const counts = Array.from({ length: buckets }, () => ({ verified: 0, review: 0, flagged: 0 }));
        let last10Verified = 0, last10Review = 0, last10Flagged = 0;
        const tenMinAgo = now.getTime() - 10 * 60 * 1000;

        for (const a of examAttempts) {
            const t = new Date(a.timestamp).getTime();
            if (Number.isNaN(t)) continue;
            if (t >= tenMinAgo) {
                const d = effectiveDecision(a);
                if (d === DEC_VERIFIED) last10Verified++;
                else if (d === DEC_REVIEW) last10Review++;
                else if (d === DEC_REJECTED) last10Flagged++;
            }
            if (t < start || t > now.getTime()) continue;
            const idx = Math.min(buckets - 1, Math.floor((t - start) / bucketMs));
            const d = effectiveDecision(a);
            if (d === DEC_VERIFIED) counts[idx].verified++;
            else if (d === DEC_REVIEW) counts[idx].review++;
            else if (d === DEC_REJECTED) counts[idx].flagged++;
        }

        const maxVal = Math.max(
            5,
            ...counts.flatMap(c => [c.verified, c.review, c.flagged])
        );
        const yMax = Math.ceil(maxVal / 5) * 5;
        renderChartAxes(yMax);

        const W = 520, H = 140;
        const xStep = W / (buckets - 1);

        const toY = (v) => H - (v / yMax) * (H - 10);
        const buildLine = (key) => {
            const pts = counts.map((c, i) => [i * xStep, toY(c[key])]);
            // smooth Bézier
            let d = `M${pts[0][0].toFixed(1)},${pts[0][1].toFixed(1)}`;
            for (let i = 1; i < pts.length; i++) {
                const [x1, y1] = pts[i - 1];
                const [x2, y2] = pts[i];
                const cx = (x1 + x2) / 2;
                d += ` C${cx.toFixed(1)},${y1.toFixed(1)} ${cx.toFixed(1)},${y2.toFixed(1)} ${x2.toFixed(1)},${y2.toFixed(1)}`;
            }
            return d;
        };
        const buildArea = (lineD) => `${lineD} L${W},${H} L0,${H} Z`;

        const verifiedLine = buildLine('verified');
        const reviewLine = buildLine('review');
        const flaggedLine = buildLine('flagged');

        const paths = svg.querySelectorAll('path');
        // Order in HTML: 3 area fills, 3 line strokes
        if (paths.length >= 6) {
            paths[0].setAttribute('d', buildArea(verifiedLine));
            paths[1].setAttribute('d', buildArea(reviewLine));
            paths[2].setAttribute('d', buildArea(flaggedLine));
            paths[3].setAttribute('d', verifiedLine);
            paths[4].setAttribute('d', reviewLine);
            paths[5].setAttribute('d', flaggedLine);
        }

        // Latest-value markers (the 3 small circles at x=520)
        const circles = svg.querySelectorAll('circle');
        if (circles.length >= 3) {
            const lastCounts = counts[counts.length - 1];
            const cy = (val, i) => {
                const y = toY(val);
                if (i < circles.length) circles[i].setAttribute('cy', String(y.toFixed(1)));
            };
            cy(lastCounts.verified, 0);
            cy(lastCounts.verified, 1); // glow circle
            cy(lastCounts.review, 2);
            if (circles.length >= 4) cy(lastCounts.flagged, 3);
        }

        // Legend totals (last 10 min)
        safeText($('bx-chart-legend-verified'), last10Verified);
        safeText($('bx-chart-legend-review'), last10Review);
        safeText($('bx-chart-legend-flagged'), last10Flagged);
    }

    // ── refresh / poll ───────────────────────────────────────────────────
    async function refresh() {
        if (typeof api === 'undefined') {
            reportError('API yüklenemedi', new Error('api.js missing'));
            return;
        }
        try {
            const snap = await api.snapshot();
            state.snapshot = snap;
            if (!state.examIdLocked || !findExam(snap, state.examId)) {
                state.examId = pickActiveExam(snap);
            }
            if (!state.examId) return;

            renderTopbar(snap, state.examId);
            renderExamPicker(snap, state.examId);
            const exam = findExam(snap, state.examId);
            renderKpis(computeKpis(snap, state.examId, exam));
            renderKpiSparklines(snap, state.examId);

            // Resolve focus BEFORE rendering surfaces that depend on it.
            // autoSelectIfNeeded sets both selectedAttemptId AND focusedStudentId.
            autoSelectIfNeeded(snap, state.examId);

            // Exam-context chips at the bottom of the student card (updated per snapshot)
            {
                const sel = state.selectedAttemptId ? findAttempt(snap, state.selectedAttemptId) : null;
                const selStudent = sel ? findStudent(snap, sel.student_id) : null;
                renderContextChips(snap, exam, selStudent);
            }

            // Simulation toolbar now sees the resolved focusedStudentId
            renderSimulationToolbar(snap, state.examId);

            renderQueue(snap, state.examId);
            renderChart(snap, state.examId);
            renderSelected(snap, state.examId);
            maybePlayNewReviewChime(snap, state.examId);

            if (!state.firstRenderDone) {
                applySkeletons(false);
                state.firstRenderDone = true;
            }

            // Refresh audit feed lazily — don't block the main render.
            // Only fetch every ~10s, or every poll if the drawer is open.
            const auditAge = Date.now() - auditState.lastFetchAt;
            if (auditState.open || auditAge > 10000) {
                loadAudit();
            }
        } catch (err) {
            reportError('Veri yüklenemedi', err);
        }
    }

    function startPolling() {
        if (state.pollHandle) return;
        state.pollHandle = setInterval(refresh, POLL_MS);
    }
    function stopPolling() {
        if (state.pollHandle) {
            clearInterval(state.pollHandle);
            state.pollHandle = null;
        }
    }

    // ── action wiring ────────────────────────────────────────────────────
    async function reviewSelected(action, label) {
        if (!state.selectedAttemptId) {
            toast('Önce bir deneme seçin.', 'error');
            return;
        }
        if (state.busy) return;
        const reasons = {
            approve: 'Manual ID check completed.',
            deny: 'Selfie did not match the enrolled reference.',
            fallback: 'Awaiting fallback ID check before access.',
        };
        state.busy = true;
        try {
            await api.reviewAttempt(state.selectedAttemptId, REVIEWER, action, reasons[action]);
            toast(label, 'success');
            const previousId = state.selectedAttemptId;
            await refresh();
            // Move on to the next reviewable attempt if one exists
            if (state.snapshot) {
                const queue = sortQueue(
                    attemptsForExam(state.snapshot, state.examId)
                        .filter(a => isOpenForReview(a) && a.attempt_id !== previousId)
                );
                if (queue.length) {
                    setFocus({ attemptId: queue[0].attempt_id, source: 'review-advance' });
                }
            }
        } catch (err) {
            reportError('İşlem başarısız', err);
        } finally {
            state.busy = false;
        }
    }

    function advanceNext() {
        if (!state.snapshot) return;
        const queue = sortQueue(
            attemptsForExam(state.snapshot, state.examId)
                .filter(a => passesFilter(a, state.snapshot, state.activeFilter))
        );
        if (queue.length === 0) return;
        const idx = queue.findIndex(a => a.attempt_id === state.selectedAttemptId);
        const next = queue[(idx + 1) % queue.length];
        setFocus({ attemptId: next.attempt_id, source: 'user-next' });
    }

    function bindActions() {
        $('bx-btn-approve')?.addEventListener('click', () => reviewSelected('approve', 'Erişim onaylandı.'));
        $('bx-btn-deny')?.addEventListener('click', () => reviewSelected('deny', 'Erişim reddedildi.'));
        $('bx-btn-defer')?.addEventListener('click', () => reviewSelected('fallback', 'Yedek kontrol istendi.'));
        $('bx-btn-next')?.addEventListener('click', advanceNext);
        $('bx-btn-refresh')?.addEventListener('click', () => refresh());
    }

    // ── Audit drawer (Feature 5) ─────────────────────────────────────────
    const LS_AUDIT_LAST_READ = 'bx-audit-last-read';
    const auditState = {
        open: false,
        lastEntries: [],     // last fetched audit_log array
        lastFetchAt: 0,
    };

    const EVENT_CLASS = {
        manual_review_completed: 'bx-audit-event-good',
        verification_attempted: 'bx-audit-event',
        demo_reset: 'bx-audit-event-warn',
        student_enrolled: 'bx-audit-event-good',
        student_preuploaded: 'bx-audit-event',
        course_saved: 'bx-audit-event',
        exam_saved: 'bx-audit-event',
        roster_imported: 'bx-audit-event',
    };

    function getAuditLastRead() {
        try { return localStorage.getItem(LS_AUDIT_LAST_READ) || ''; }
        catch (e) { return ''; }
    }
    function setAuditLastRead(ts) {
        try { localStorage.setItem(LS_AUDIT_LAST_READ, ts || ''); } catch (e) {}
    }

    function fmtAuditTime(ts) {
        if (!ts) return '—';
        const d = new Date(ts);
        if (Number.isNaN(d.getTime())) return ts;
        return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
    }
    function fmtAuditDate(ts) {
        if (!ts) return '';
        const d = new Date(ts);
        if (Number.isNaN(d.getTime())) return '';
        return `${d.getDate()} ${TR_MONTHS[d.getMonth()]}`;
    }

    function renderAuditList(entries) {
        const ul = $('bx-audit-list');
        if (!ul) return;
        if (!entries || entries.length === 0) {
            ul.innerHTML = '<li class="bx-audit-empty">Henüz denetim kaydı yok.</li>';
            return;
        }
        const sorted = entries.slice().sort((a, b) =>
            new Date(b.timestamp || 0) - new Date(a.timestamp || 0)
        );
        const eventLabelFn = window.eventLabel || ((s) => s);
        ul.innerHTML = sorted.slice(0, 80).map(e => {
            const klass = EVENT_CLASS[e.event_type] || 'bx-audit-event';
            return `
                <li>
                    <div class="bx-audit-head-row">
                        <span class="${klass}">${escapeHtml(eventLabelFn(e.event_type))}</span>
                        <span class="bx-audit-time">${escapeHtml(fmtAuditDate(e.timestamp))} ${escapeHtml(fmtAuditTime(e.timestamp))}</span>
                    </div>
                    <div class="bx-audit-actor">${escapeHtml(e.actor || '—')}</div>
                    <span class="bx-audit-message">${escapeHtml(e.message || '')}</span>
                </li>
            `;
        }).join('');
    }

    async function loadAudit() {
        try {
            const resp = await api.audit();
            const list = resp.audit_log || resp || [];
            auditState.lastEntries = list;
            auditState.lastFetchAt = Date.now();
            renderAuditList(list);
            updateAuditBadge();
        } catch (err) {
            reportError('Denetim yüklenemedi', err);
        }
    }

    function updateAuditBadge() {
        const bell = $('bx-bell-count');
        if (!bell) return;
        // The bell badge dual-purposes:
        //  - Until first audit fetch: show review-queue count (legacy behavior)
        //  - After first audit fetch: show unread audit-log entries since last drawer open
        if (auditState.lastEntries.length === 0) return; // keep legacy behavior
        const lastRead = getAuditLastRead();
        const unread = auditState.lastEntries.filter(e =>
            !lastRead || (e.timestamp || '') > lastRead
        ).length;
        bell.textContent = String(unread);
        bell.style.display = unread ? '' : 'none';
        // Live flash if unread count changed
        flashChanged(bell, unread, 'bx-bell-count-audit');
    }

    function openAuditDrawer() {
        const drawer = $('bx-audit-drawer');
        const backdrop = $('bx-audit-backdrop');
        if (!drawer) return;
        auditState.open = true;
        drawer.classList.add('is-open');
        drawer.setAttribute('aria-hidden', 'false');
        backdrop?.classList.add('is-open');
        // Mark all current entries read
        const newest = auditState.lastEntries
            .map(e => e.timestamp || '')
            .reduce((a, b) => (a > b ? a : b), '');
        if (newest) setAuditLastRead(newest);
        updateAuditBadge();
        // Force a fresh fetch on open — bypass the 10s throttle so the drawer
        // always shows the latest audit entries, not whatever was cached.
        auditState.lastFetchAt = 0;
        loadAudit();
    }
    function closeAuditDrawer() {
        const drawer = $('bx-audit-drawer');
        const backdrop = $('bx-audit-backdrop');
        auditState.open = false;
        drawer?.classList.remove('is-open');
        drawer?.setAttribute('aria-hidden', 'true');
        backdrop?.classList.remove('is-open');
    }

    function bindAuditDrawer() {
        // Bell button is the first .bx-bell in topbar
        const bell = document.querySelector('.bx-icon-btn.bx-bell');
        bell?.addEventListener('click', (e) => {
            e.preventDefault();
            auditState.open ? closeAuditDrawer() : openAuditDrawer();
        });
        $('bx-audit-close')?.addEventListener('click', closeAuditDrawer);
        $('bx-audit-backdrop')?.addEventListener('click', closeAuditDrawer);
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && auditState.open) closeAuditDrawer();
        });
    }

    // ── Feature Match re-verify (Feature 11) ─────────────────────────────
    const reverifyCache = new Map();  // attempt_id → { score, decision, threshold }

    async function dataUrlToBlob(dataUrl) {
        const resp = await fetch(dataUrl);
        return await resp.blob();
    }

    function showReverifyBadge(text, kind) {
        const badge = $('bx-reverify-badge');
        if (!badge) return;
        badge.textContent = text;
        badge.classList.remove('is-good', 'is-bad', 'is-warn');
        if (kind) badge.classList.add('is-' + kind);
        badge.hidden = false;
    }
    function hideReverifyBadge() {
        const badge = $('bx-reverify-badge');
        if (badge) badge.hidden = true;
    }

    function applyCachedReverify(attemptId) {
        if (!attemptId) { hideReverifyBadge(); return; }
        const cached = reverifyCache.get(attemptId);
        if (!cached) { hideReverifyBadge(); return; }
        const pct = (cached.score * 100).toFixed(1);
        const kind = cached.decision === DEC_VERIFIED ? 'good'
                   : cached.decision === DEC_REJECTED ? 'bad' : 'warn';
        const labelMap = {
            verified: 'Doğrulandı',
            manual_review: 'İnceleme',
            rejected: 'Reddedildi',
        };
        showReverifyBadge(`${labelMap[cached.decision] || cached.decision} · %${pct}`, kind);
    }

    async function reverifySelected() {
        const attempt = state.selectedAttemptId
            ? findAttempt(state.snapshot, state.selectedAttemptId)
            : null;
        const exam = findExam(state.snapshot, state.examId);
        const student = attempt ? findStudent(state.snapshot, attempt.student_id) : null;
        if (!attempt || !exam || !student) {
            toast('Önce bir deneme seçin.', 'error');
            return;
        }
        if (!attempt.query_preview || !student.reference_preview) {
            toast('Doğrulama için görüntüler eksik.', 'error');
            return;
        }
        const btn = $('bx-reverify-btn');
        btn?.classList.add('is-busy');
        try {
            const [qBlob, rBlob] = await Promise.all([
                dataUrlToBlob(attempt.query_preview),
                dataUrlToBlob(student.reference_preview),
            ]);
            // Convert blobs to File objects so multipart upload names are stable
            const qFile = new File([qBlob], 'query.jpg', { type: qBlob.type || 'image/jpeg' });
            const rFile = new File([rBlob], 'enrolled.jpg', { type: rBlob.type || 'image/jpeg' });
            const result = await api.modelLabCompare(exam.model_type, qFile, rFile);
            reverifyCache.set(attempt.attempt_id, {
                score: Number(result.score) || 0,
                decision: result.decision || 'manual_review',
                threshold: Number(result.threshold) || 0,
            });
            applyCachedReverify(attempt.attempt_id);
            toast('Yeniden doğrulama tamamlandı.', 'success');
        } catch (err) {
            reportError('Yeniden doğrulama başarısız', err);
        } finally {
            btn?.classList.remove('is-busy');
        }
    }

    function bindReverify() {
        $('bx-reverify-btn')?.addEventListener('click', reverifySelected);
    }

    // ── Real Diff Map (Feature 10) ───────────────────────────────────────
    const diffCache = new Map();  // attempt_id → data URL of computed diff

    function loadImage(src) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            // data: URLs work cross-origin; HTTPS images on same origin also OK
            img.crossOrigin = 'anonymous';
            img.onload = () => resolve(img);
            img.onerror = (e) => reject(new Error('image load failed'));
            img.src = src;
        });
    }

    async function computeDiffDataUrl(queryUrl, refUrl) {
        const SIZE = 220;
        const [qImg, rImg] = await Promise.all([loadImage(queryUrl), loadImage(refUrl)]);
        const canvas = document.createElement('canvas');
        canvas.width = canvas.height = SIZE;
        const ctx = canvas.getContext('2d');

        // Draw query and capture
        ctx.drawImage(qImg, 0, 0, SIZE, SIZE);
        const qData = ctx.getImageData(0, 0, SIZE, SIZE);

        // Draw reference and capture
        ctx.drawImage(rImg, 0, 0, SIZE, SIZE);
        const rData = ctx.getImageData(0, 0, SIZE, SIZE);

        const out = ctx.createImageData(SIZE, SIZE);
        const q = qData.data;
        const r = rData.data;
        const o = out.data;
        for (let i = 0; i < q.length; i += 4) {
            const dr = Math.abs(q[i] - r[i]);
            const dg = Math.abs(q[i + 1] - r[i + 1]);
            const db = Math.abs(q[i + 2] - r[i + 2]);
            const intensity = (dr + dg + db) / 3;
            if (intensity < 28) {
                o[i] = 16;  o[i + 1] = 185; o[i + 2] = 129;
            } else if (intensity < 70) {
                o[i] = 245; o[i + 1] = 158; o[i + 2] = 11;
            } else {
                o[i] = 239; o[i + 1] = 68;  o[i + 2] = 68;
            }
            o[i + 3] = 200;
        }
        ctx.putImageData(out, 0, 0);
        return canvas.toDataURL('image/png');
    }

    async function renderDiffMap(attempt, student) {
        if (!attempt) return;
        const img = $('bx-evidence-diff');
        if (!img) return;
        const queryUrl = attempt.query_preview;
        const refUrl = (student && student.reference_preview) || queryUrl;
        if (!queryUrl || !refUrl) return;

        // Skip if same image (no reference enrolled): keep decorative pattern
        if (queryUrl === refUrl) return;

        const key = attempt.attempt_id;
        const tile = img.closest('.bx-evidence-img');

        // Cached? use immediately
        if (diffCache.has(key)) {
            img.src = diffCache.get(key);
            tile?.classList.add('bx-diff-computed');
            return;
        }
        try {
            const dataUrl = await computeDiffDataUrl(queryUrl, refUrl);
            diffCache.set(key, dataUrl);
            // Only apply if user hasn't navigated away
            if (state.selectedAttemptId === key) {
                img.src = dataUrl;
                tile?.classList.add('bx-diff-computed');
            }
        } catch (e) {
            // Silently keep the decorative overlay if image loading failed
            console.warn('diff map failed', e);
        }
    }

    // ── Simulation trigger (Feature 12) ──────────────────────────────────
    const simState = {
        scenario: 'matching',
        studentId: null,
        consent: false,
        stagedFile: null,
        busy: false,
        userPickedStudent: false,  // true if user manually changed the <select>
    };

    function pickScenarioTargetStudent(scenario, snapshot, examId) {
        if (!snapshot) return { studentId: null, error: 'snapshot eksik' };
        const exam = findExam(snapshot, examId);
        const roster = exam ? rosterForExam(snapshot, exam) : (snapshot.students || []);
        if (roster.length === 0) return { studentId: null, error: 'Listede öğrenci yok' };

        const findEnrolled = () => {
            // Prefer Aylin Kaya if present
            const aylin = roster.find(s => /aylin/i.test(s.name || '') && Number(s.sample_count || 0) > 0);
            if (aylin) return aylin;
            return roster.find(s => Number(s.sample_count || 0) > 0) || null;
        };

        if (scenario === 'matching' || scenario === 'needs_review') {
            const target = findEnrolled();
            if (!target) return { studentId: null, error: 'Kayıtlı öğrenci yok; önce kayıt yapın.' };
            return { studentId: target.student_id, label: target.name };
        }
        if (scenario === 'no_enrollment') {
            const target = roster.find(s => Number(s.sample_count || 0) === 0);
            if (!target) return { studentId: null, error: 'Tüm öğrenciler kayıtlı; demoyu sıfırlayın.' };
            return { studentId: target.student_id, label: target.name };
        }
        return { studentId: null, error: 'Bilinmeyen senaryo' };
    }

    function renderSimulationToolbar(snapshot, examId) {
        const select = $('bx-sim-student');
        const fileLabelEl = $('bx-sim-file-label');
        const fileLabelHost = document.querySelector('.bx-sim-file');
        const consentBox = $('bx-sim-consent');
        const submitPreloaded = $('bx-sim-submit-preloaded');
        const submitUpload = $('bx-sim-submit-upload');
        const hint = $('bx-sim-hint');
        const chips = document.querySelectorAll('.bx-sim-chip[data-scenario]');
        if (!select || !submitPreloaded) return;

        // Sync chip active state
        chips.forEach(b => {
            const active = b.dataset.scenario === simState.scenario;
            b.classList.toggle('is-active', active);
            b.setAttribute('aria-selected', active ? 'true' : 'false');
        });

        // Populate student dropdown from roster
        const exam = findExam(snapshot, examId);
        const roster = exam ? rosterForExam(snapshot, exam) : (snapshot?.students || []);
        const sorted = roster.slice().sort((a, b) => (a.name || '').localeCompare(b.name || '', 'tr'));

        // Follow state.focusedStudentId as the single source of truth.
        // If no focus is set yet, fall back to the scenario default.
        if (state.focusedStudentId && sorted.find(s => s.student_id === state.focusedStudentId)) {
            simState.studentId = state.focusedStudentId;
        } else if (!simState.studentId || !sorted.find(s => s.student_id === simState.studentId)) {
            const target = pickScenarioTargetStudent(simState.scenario, snapshot, examId);
            if (target.studentId) simState.studentId = target.studentId;
        }

        // Repopulate options if they changed
        const currentOptions = [...select.options].map(o => o.value).join('|');
        const wantOptions = sorted.map(s => s.student_id).join('|');
        if (currentOptions !== wantOptions) {
            select.innerHTML = '';
            if (sorted.length === 0) {
                const opt = document.createElement('option');
                opt.value = '';
                opt.textContent = 'Öğrenci yok';
                select.appendChild(opt);
            } else {
                for (const s of sorted) {
                    const opt = document.createElement('option');
                    opt.value = s.student_id;
                    const sampleHint = Number(s.sample_count || 0) > 0 ? '' : ' · Kayıt yok';
                    opt.textContent = `${s.name} (${s.student_id})${sampleHint}`;
                    select.appendChild(opt);
                }
            }
        }
        // Don't clobber the dropdown if the user has it open / focused
        if (simState.studentId && select.value !== simState.studentId && !isUserEditing(select)) {
            select.value = simState.studentId;
        }

        // Sync consent + file label (skip checkbox write if user has focus on it
        // — they might be mid-toggle and the change handler hasn't fired yet)
        if (consentBox && !isUserEditing(consentBox) && consentBox.checked !== simState.consent) {
            consentBox.checked = simState.consent;
        }
        if (fileLabelEl) {
            fileLabelEl.textContent = simState.stagedFile
                ? `Yüklü: ${simState.stagedFile.name}`
                : 'Selfie yükle…';
        }
        if (fileLabelHost) {
            fileLabelHost.classList.toggle('has-file', !!simState.stagedFile);
        }

        // Hint text
        let hintText = '';
        let hintClass = '';
        if (!exam) {
            hintText = 'Sınav yükleniyor…';
            hintClass = '';
        } else if (simState.stagedFile) {
            hintText = `Yüklenen dosya ile doğrulanacak (${simState.scenario === 'needs_review' ? 'impostor' : 'matching'})`;
            hintClass = '';
        } else if (simState.scenario === 'matching') {
            hintText = 'Hazır eşleşen FLUXSynID selfie';
        } else if (simState.scenario === 'needs_review') {
            hintText = 'Hazır impostor selfie (manuel inceleme oluşturur)';
            hintClass = 'is-warn';
        } else if (simState.scenario === 'no_enrollment') {
            hintText = 'Önce öğrenci kaydı gerekli';
            hintClass = 'is-warn';
        }
        if (hint) {
            hint.textContent = hintText;
            hint.classList.remove('is-warn', 'is-error');
            if (hintClass) hint.classList.add(hintClass);
        }

        // Enable/disable buttons
        const hasContext = !!exam && !!simState.studentId;
        const canPreloaded = hasContext && simState.consent && simState.scenario !== 'no_enrollment' && !simState.busy;
        const canUpload = hasContext && simState.consent && !!simState.stagedFile && !simState.busy;
        submitPreloaded.disabled = !canPreloaded;
        submitUpload.disabled = !canUpload;
    }

    async function submitSimulation(kind) {
        if (simState.busy) return;
        if (!simState.consent) { toast('Önce rıza onayı gerekli.', 'error'); return; }
        if (!simState.studentId || !state.examId) {
            toast('Sınav ve öğrenci seçin.', 'error');
            return;
        }
        simState.busy = true;
        renderSimulationToolbar(state.snapshot, state.examId);
        try {
            let result;
            if (kind === 'upload') {
                if (!simState.stagedFile) { toast('Önce bir selfie dosyası seçin.', 'error'); return; }
                result = await api.verifyStudent(state.examId, simState.studentId, simState.stagedFile);
            } else {
                const backendScenario = simState.scenario === 'needs_review' ? 'impostor' : 'matching';
                result = await api.verifyPreloadedStudent(
                    state.examId, simState.studentId, backendScenario, true
                );
            }
            const newId = result?.attempt?.attempt_id;
            const decisionTr = (window.decisionLabel || ((d) => d))(result?.attempt?.decision || '');
            toast(`Deneme oluşturuldu: ${decisionTr}`, 'success');

            // Switch to "Tümü" so any decision is visible, then auto-focus on the
            // new attempt. refresh() fetches the latest snapshot before setFocus
            // can locate the attempt. setFilter handles the tab visual atomically.
            setFilter('all');
            await refresh();
            if (newId) setFocus({ attemptId: newId, source: 'sim-submit' });

            // Scroll the queue card into view so the new row is visible
            document.querySelector('.bx-right-col .bx-card')
                ?.scrollIntoView({ behavior: 'smooth', block: 'start' });
        } catch (err) {
            reportError('Simülasyon başarısız', err);
        } finally {
            simState.busy = false;
            renderSimulationToolbar(state.snapshot, state.examId);
        }
    }

    function bindSimulationToolbar() {
        document.querySelectorAll('.bx-sim-chip[data-scenario]').forEach(btn => {
            btn.addEventListener('click', () => {
                simState.scenario = btn.dataset.scenario;
                // Re-pick target student for new scenario, then sync focus
                const target = pickScenarioTargetStudent(simState.scenario, state.snapshot, state.examId);
                if (target.studentId) {
                    setFocus({ studentId: target.studentId, source: 'user-toolbar' });
                } else {
                    if (target.error) toast(target.error, 'error');
                    // Re-render to update hint/buttons for the new scenario even if student unchanged
                    renderSimulationToolbar(state.snapshot, state.examId);
                }
            });
        });
        $('bx-sim-student')?.addEventListener('change', (e) => {
            setFocus({ studentId: e.target.value || null, source: 'user-toolbar' });
        });
        $('bx-sim-consent')?.addEventListener('change', (e) => {
            simState.consent = !!e.target.checked;
            renderSimulationToolbar(state.snapshot, state.examId);
        });
        $('bx-sim-file')?.addEventListener('change', (e) => {
            simState.stagedFile = (e.target.files && e.target.files[0]) || null;
            renderSimulationToolbar(state.snapshot, state.examId);
        });
        $('bx-sim-submit-preloaded')?.addEventListener('click', () => submitSimulation('preloaded'));
        $('bx-sim-submit-upload')?.addEventListener('click', () => submitSimulation('upload'));
    }

    // ── Polish round: helpers (close, view-log, help, sidebar, chips, steps, axes, hero) ──
    const HELP_URL = 'https://github.com/AtakanEcevit/LAP-2526#readme';

    const MODEL_LABELS = {
        facenet_contrastive_proto: 'FaceNet Contrastive Proto',
        facenet_contrastive_proto_model5: 'FaceNet Contrastive Proto Model 5',
        facenet_arcface_triplet_model6: 'FaceNet ArcFace Triplet Model 6',
        facenet_hybrid: 'Hybrid FaceNet',
        facenet_proto: 'FaceNet Proto',
        facenet: 'FaceNet',
    };

    function humanizeModel(raw) {
        if (!raw) return '—';
        const lower = String(raw).toLowerCase();
        return MODEL_LABELS[lower] || raw;
    }

    // Section 1: detail close
    function bindCloseDetail() {
        $('bx-detail-close')?.addEventListener('click', (e) => {
            e.preventDefault();
            setFocus({ studentId: null, attemptId: null, source: 'user-close' });
        });
    }

    // Section 1: "Tüm Günlüğü Gör" → open audit drawer
    function bindAuditLink() {
        $('bx-view-full-log')?.addEventListener('click', (e) => {
            e.preventDefault();
            openAuditDrawer();
        });
    }

    // Section 2: help button → README in new tab
    function bindHelpLink() {
        $('bx-help-btn')?.addEventListener('click', (e) => {
            e.preventDefault();
            window.open(HELP_URL, '_blank', 'noopener,noreferrer');
        });
    }

    // Section 3: sidebar nav cross-wiring
    function bindSidebarNav() {
        const scrollTo = (selector) => {
            const el = document.querySelector(selector);
            if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        };

        const setActiveNav = (id) => {
            document.querySelectorAll('.bx-sidebar .bx-nav-link').forEach(a => {
                a.classList.toggle('is-active', a.id === id);
            });
        };
        const revertActiveSoon = () => {
            // For toast-only handlers, revert the highlighted link back to Kontrol Paneli after a brief delay
            setTimeout(() => setActiveNav('bx-nav-kontrol-paneli'), 1100);
        };

        const setQueueFilter = (filter) => setFilter(filter);

        const bindLink = (id, handler) => {
            const el = $(id);
            if (!el) return;
            el.addEventListener('click', (e) => {
                e.preventDefault();
                handler();
            });
        };

        bindLink('bx-nav-kontrol-paneli', () => window.scrollTo({ top: 0, behavior: 'smooth' }));
        bindLink('bx-nav-sinav-kapisi', () => {
            toast('Sınav kapısı ana uygulamadan yönetiliyor.');
            revertActiveSoon();
        });
        bindLink('bx-nav-canli-izleme', () => scrollTo('.bx-card-student'));
        bindLink('bx-nav-sinav-oturumlari', () => scrollTo('.bx-right-col .bx-card'));
        bindLink('bx-nav-inceleme', () => {
            setQueueFilter('review');
            scrollTo('.bx-right-col .bx-card');
        });
        bindLink('bx-nav-isaretlenen', () => {
            setQueueFilter('flagged');
            scrollTo('.bx-right-col .bx-card');
        });
        bindLink('bx-nav-denetim-kayitlari', () => {
            openAuditDrawer();
            revertActiveSoon();
        });

        // Settings group — toast and revert
        ['bx-nav-sinav-ayarlari', 'bx-nav-dogrulama-kurallari', 'bx-nav-kullanicilar', 'bx-nav-entegrasyonlar'].forEach(id => {
            bindLink(id, () => {
                toast('Yakında — bu özellik için ayar paneli geliyor.');
                revertActiveSoon();
            });
        });
    }

    // Section 4: exam-context chips
    function renderContextChips(snapshot, exam, student) {
        // Model
        const modelEl = $('bx-chip-model-value');
        safeText(modelEl, exam ? humanizeModel(exam.model_type) : '—');

        // Eşik
        const thresholdEl = $('bx-chip-threshold-value');
        safeText(thresholdEl, (exam && Number.isFinite(exam.threshold))
            ? Number(exam.threshold).toFixed(3)
            : '—');

        // Pencere
        const windowEl = $('bx-chip-window-value');
        if (exam && exam.start_time && exam.end_time) {
            const fmt = window.formatTime || ((iso) => {
                const d = new Date(iso);
                if (Number.isNaN(d.getTime())) return '—';
                return `${pad(d.getHours())}:${pad(d.getMinutes())}`;
            });
            safeText(windowEl, `${fmt(exam.start_time)} – ${fmt(exam.end_time)}`);
        } else {
            safeText(windowEl, '—');
        }

        // Kayıt — selected student's sample count, or exam-roster aggregate
        const enrollmentEl = $('bx-chip-enrollment-value');
        if (student) {
            const count = Number(student.sample_count || 0);
            safeText(enrollmentEl, count > 0 ? `${count} örnek` : 'Kayıt yok');
        } else if (exam) {
            const roster = rosterForExam(snapshot, exam);
            const enrolled = roster.filter(s => Number(s.sample_count || 0) > 0).length;
            safeText(enrollmentEl, `${enrolled} / ${roster.length}`);
        } else {
            safeText(enrollmentEl, '—');
        }
    }

    // Section 5: steps rail reflects attempt state
    const CHECK_SVG = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>';
    const X_SVG = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18M6 6l12 12"/></svg>';

    function setStepState(li, state, label) {
        if (!li) return;
        li.classList.remove('done', 'active', 'bx-step-bad');
        if (state === 'done') li.classList.add('done');
        else if (state === 'done-bad') li.classList.add('done', 'bx-step-bad');
        else if (state === 'active') li.classList.add('active');
        // mark content
        const mark = li.querySelector('.bx-step-mark');
        if (mark) {
            if (state === 'done') mark.innerHTML = CHECK_SVG;
            else if (state === 'done-bad') mark.innerHTML = X_SVG;
            else mark.textContent = label;
        }
    }

    function renderSteps(attempt, student) {
        const consent = $('bx-step-consent');
        const enrollment = $('bx-step-enrollment');
        const verify = $('bx-step-verify');
        const result = $('bx-step-result');

        // No attempt: show enrollment-readiness only
        if (!attempt) {
            setStepState(consent, student ? 'done' : 'active', '1');
            const hasEnroll = student && Number(student.sample_count || 0) > 0;
            setStepState(enrollment, hasEnroll ? 'done' : (student ? 'active' : 'pending'), '2');
            setStepState(verify, 'pending', '3');
            setStepState(result, 'pending', '4');
            return;
        }

        // Attempt exists → consent + enrollment implicitly done
        setStepState(consent, 'done', '1');
        const hasEnroll = student && Number(student.sample_count || 0) > 0;
        setStepState(enrollment, hasEnroll ? 'done' : 'done', '2');  // attempt presumes enrollment

        // Verify step: done when the attempt has a decision (it always does after recording)
        setStepState(verify, 'done', '3');

        // Result step: depends on final_status
        const fs = attempt.final_status || attempt.status;
        if (fs === 'Approved by Proctor' || attempt.decision === 'verified') {
            setStepState(result, 'done', '4');
        } else if (fs === 'Rejected' || attempt.decision === 'rejected') {
            setStepState(result, 'done-bad', '4');
        } else {
            // Manual Review / Fallback Requested → still active
            setStepState(result, 'active', '4');
        }
    }

    // Section 6: dynamic chart axis labels
    function renderChartAxes(yMax) {
        // Y-axis: 4 ticks top-to-bottom: yMax, 2/3·yMax, 1/3·yMax, 0
        safeText($('bx-chart-y-3'), String(yMax));
        safeText($('bx-chart-y-2'), String(Math.round(yMax * 2 / 3)));
        safeText($('bx-chart-y-1'), String(Math.round(yMax / 3)));
        safeText($('bx-chart-y-0'), '0');

        // X-axis: 6 ticks across the last 60 min in 10-min steps + "now"
        const now = new Date();
        const buckets = 6;
        const bucketMs = 10 * 60 * 1000;
        const start = now.getTime() - (buckets - 1) * bucketMs;
        const fmt = (t) => {
            const d = new Date(t);
            return `${pad(d.getHours())}:${pad(d.getMinutes())}`;
        };
        for (let i = 0; i < buckets; i++) {
            const el = $(`bx-chart-x-${i}`);
            safeText(el, fmt(start + i * bucketMs));
        }
    }

    // Section 7: hero caption reflects selected attempt
    const HERO_VARIANTS = {
        verified:       { title: 'Yüz doğrulandı',         iconClass: '',                       icon: CHECK_SVG },
        manual_review:  { title: 'Manuel inceleme gerekli', iconClass: 'bx-check-circle-warn',  icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M12 9v4"/><path d="M12 17h.01"/></svg>' },
        rejected:       { title: 'Eşleşme reddedildi',     iconClass: 'bx-check-circle-bad',   icon: X_SVG },
        none:           { title: 'Deneme bekleniyor',      iconClass: 'bx-check-circle-muted', icon: '<svg viewBox="0 0 24 24" fill="currentColor"><circle cx="6" cy="12" r="1.6"/><circle cx="12" cy="12" r="1.6"/><circle cx="18" cy="12" r="1.6"/></svg>' },
    };

    function renderHeroCaption(attempt, student) {
        const icon = $('bx-hero-cap-icon');
        const svgHost = $('bx-hero-cap-svg');
        const titleEl = $('bx-hero-cap-title');
        const subEl = $('bx-hero-cap-sub');
        if (!icon || !titleEl || !subEl) return;

        let kind;
        if (!attempt) kind = 'none';
        else {
            const d = effectiveDecision(attempt);
            if (d === DEC_VERIFIED) kind = 'verified';
            else if (d === DEC_REJECTED) kind = 'rejected';
            else kind = 'manual_review';
        }
        const variant = HERO_VARIANTS[kind];

        // Update icon container classes
        icon.classList.remove('bx-check-circle-warn', 'bx-check-circle-bad', 'bx-check-circle-muted');
        if (variant.iconClass) icon.classList.add(variant.iconClass);

        // Replace SVG content while preserving the id="bx-hero-cap-svg" anchor for future renders
        icon.innerHTML = variant.icon.replace('<svg', '<svg id="bx-hero-cap-svg"');

        safeText(titleEl, variant.title);
        safeText(subEl, (attempt && (attempt.student_name || (student && student.name))) || '—');
    }

    // ── Sound notification (Feature 9) ───────────────────────────────────
    const LS_SOUND_KEY = 'bx-sound-enabled';
    const soundState = {
        enabled: false,
        ctx: null,    // AudioContext (lazy)
    };

    function loadSoundPreference() {
        try { return localStorage.getItem(LS_SOUND_KEY) === 'true'; }
        catch (e) { return false; }
    }
    function persistSoundPreference(on) {
        try { localStorage.setItem(LS_SOUND_KEY, on ? 'true' : 'false'); } catch (e) {}
    }

    function ensureAudioContext() {
        if (soundState.ctx) return soundState.ctx;
        const AC = window.AudioContext || window.webkitAudioContext;
        if (!AC) return null;
        soundState.ctx = new AC();
        return soundState.ctx;
    }

    function playChime() {
        if (!soundState.enabled) return;
        const ctx = ensureAudioContext();
        if (!ctx) return;
        try {
            if (ctx.state === 'suspended') ctx.resume();
        } catch (e) {}
        const now = ctx.currentTime;
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(880, now);
        osc.frequency.exponentialRampToValueAtTime(1320, now + 0.18);
        gain.gain.setValueAtTime(0.0001, now);
        gain.gain.exponentialRampToValueAtTime(0.18, now + 0.05);
        gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.55);
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start(now);
        osc.stop(now + 0.6);
    }

    function updateSoundIcons() {
        const on = $('bx-sound-icon-on');
        const off = $('bx-sound-icon-off');
        const btn = $('bx-sound-toggle');
        if (!on || !off || !btn) return;
        on.hidden = !soundState.enabled;
        off.hidden = soundState.enabled;
        btn.setAttribute('aria-pressed', soundState.enabled ? 'true' : 'false');
        btn.setAttribute('aria-label', soundState.enabled ? 'Ses açık' : 'Ses kapalı');
    }

    function bindSoundToggle() {
        const btn = $('bx-sound-toggle');
        if (!btn) return;
        soundState.enabled = loadSoundPreference();
        updateSoundIcons();
        btn.addEventListener('click', () => {
            soundState.enabled = !soundState.enabled;
            persistSoundPreference(soundState.enabled);
            updateSoundIcons();
            // Initialize AudioContext on user gesture so chimes can play later
            if (soundState.enabled) {
                ensureAudioContext();
                playChime();  // confirm sound
                toast('Ses bildirimi açık.', 'success');
            } else {
                toast('Ses bildirimi kapalı.');
            }
        });
    }

    // Detect new manual_review attempts arriving between polls.
    function maybePlayNewReviewChime(snapshot, examId) {
        if (!soundState.enabled) {
            // Still track IDs so we don't fire a flood when sound is later enabled
            state.lastReviewIds = new Set(
                attemptsForExam(snapshot, examId).filter(isOpenForReview).map(a => a.attempt_id)
            );
            return;
        }
        const currentIds = new Set(
            attemptsForExam(snapshot, examId).filter(isOpenForReview).map(a => a.attempt_id)
        );
        if (state.lastReviewIds !== null) {
            let isNew = false;
            for (const id of currentIds) {
                if (!state.lastReviewIds.has(id)) { isNew = true; break; }
            }
            if (isNew) playChime();
        }
        state.lastReviewIds = currentIds;
    }

    // ── KPI sparklines (Feature 8) ───────────────────────────────────────
    function renderKpiSparklines(snapshot, examId) {
        const examAttempts = attemptsForExam(snapshot, examId);
        const buckets = 6;
        const bucketMs = 5 * 60 * 1000;  // 5 min
        const now = Date.now();
        const start = now - buckets * bucketMs;

        const series = {
            total: new Array(buckets).fill(0),
            verified: new Array(buckets).fill(0),
            review: new Array(buckets).fill(0),
            flagged: new Array(buckets).fill(0),
        };
        for (const a of examAttempts) {
            const t = new Date(a.timestamp).getTime();
            if (Number.isNaN(t)) continue;
            if (t < start || t > now) continue;
            const idx = Math.min(buckets - 1, Math.floor((t - start) / bucketMs));
            series.total[idx]++;
            const d = effectiveDecision(a);
            if (d === DEC_VERIFIED) series.verified[idx]++;
            else if (d === DEC_REVIEW) series.review[idx]++;
            else if (d === DEC_REJECTED) series.flagged[idx]++;
        }

        const buildPath = (vals, W, H) => {
            const max = Math.max(1, ...vals);
            const xStep = W / (buckets - 1);
            const toY = (v) => H - (v / max) * (H - 2) - 1;
            let d = '';
            for (let i = 0; i < vals.length; i++) {
                const x = (i * xStep).toFixed(1);
                const y = toY(vals[i]).toFixed(1);
                if (i === 0) d += `M${x},${y}`;
                else d += ` L${x},${y}`;
            }
            return d;
        };
        const W = 60, H = 18;
        const apply = (id, key) => {
            const svg = $(id);
            if (!svg) return;
            const path = svg.querySelector('path');
            if (path) path.setAttribute('d', buildPath(series[key], W, H));
        };
        apply('bx-spark-total', 'total');
        apply('bx-spark-verified', 'verified');
        apply('bx-spark-review', 'review');
        apply('bx-spark-flagged', 'flagged');
        // Kayıt Yok: roster-derived, not time-bucketed. Use a flat line at mid.
        const noregSvg = $('bx-spark-noreg');
        if (noregSvg) {
            const path = noregSvg.querySelector('path');
            if (path) path.setAttribute('d', `M0,${H/2} L${W},${H/2}`);
        }
    }

    // ── Queue search + sort (Feature 6) ──────────────────────────────────
    const SORT_LABELS = {
        priority: 'Öncelik',
        newest: 'En yeni',
        oldest: 'En eski',
        'score-asc': 'Skor ↑',
        'score-desc': 'Skor ↓',
    };

    function hydrateSortModeFromStorage() {
        try {
            const stored = localStorage.getItem(LS_SORT_KEY);
            if (stored && SORT_LABELS[stored]) state.sortMode = stored;
        } catch (e) {}
    }
    function persistSortMode(mode) {
        try { localStorage.setItem(LS_SORT_KEY, mode); } catch (e) {}
    }

    function updateSortMenuSelection() {
        const menu = $('bx-sort-menu');
        if (!menu) return;
        menu.querySelectorAll('li[data-sort]').forEach(li => {
            li.classList.toggle('is-selected', li.dataset.sort === state.sortMode);
        });
        const lbl = $('bx-sort-label');
        if (lbl) lbl.textContent = SORT_LABELS[state.sortMode] || 'Öncelik';
    }

    function bindSortMenu() {
        const btn = $('bx-sort-toggle');
        const menu = $('bx-sort-menu');
        if (!btn || !menu) return;
        const close = () => { menu.hidden = true; btn.setAttribute('aria-expanded', 'false'); };
        const open = () => { menu.hidden = false; btn.setAttribute('aria-expanded', 'true'); };
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            menu.hidden ? open() : close();
        });
        menu.addEventListener('click', (e) => {
            const li = e.target.closest('li[data-sort]');
            if (!li) return;
            state.sortMode = li.dataset.sort;
            state.page = 1;
            persistSortMode(state.sortMode);
            updateSortMenuSelection();
            close();
            if (state.snapshot && state.examId) renderQueue(state.snapshot, state.examId);
        });
        document.addEventListener('click', (e) => {
            if (menu.hidden) return;
            if (btn.contains(e.target) || menu.contains(e.target)) return;
            close();
        });
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && !menu.hidden) close();
        });
        updateSortMenuSelection();
    }

    function bindQueueSearch() {
        const input = $('bx-queue-search');
        if (!input) return;
        let debounceHandle = null;
        input.addEventListener('input', () => {
            clearTimeout(debounceHandle);
            debounceHandle = setTimeout(() => {
                state.searchTerm = input.value || '';
                state.page = 1;
                if (state.snapshot && state.examId) {
                    autoSelectIfNeeded(state.snapshot, state.examId);
                    renderQueue(state.snapshot, state.examId);
                    renderSelected(state.snapshot, state.examId);
                }
            }, 100);
        });
    }

    // ── Exam picker (Feature 4) ──────────────────────────────────────────
    const LS_EXAM_KEY = 'bx-selected-exam-id';

    function hydrateExamSelectionFromStorage() {
        try {
            const stored = localStorage.getItem(LS_EXAM_KEY);
            if (stored && stored !== 'auto') {
                state.examId = stored;
                state.examIdLocked = true;
            }
        } catch (e) { /* localStorage may be unavailable */ }
    }

    function persistExamSelection(value) {
        try { localStorage.setItem(LS_EXAM_KEY, value); } catch (e) {}
    }

    function renderExamPicker(snapshot, examId) {
        const label = $('bx-exam-picker-label');
        const menu = $('bx-exam-menu');
        if (!label || !menu) return;

        const exam = findExam(snapshot, examId);
        const examName = exam ? (exam.name || examId) : 'Otomatik';
        label.textContent = state.examIdLocked ? examName : `Otomatik: ${examName}`;

        // Build menu: Auto + each exam with its attempt count for that exam
        const exams = snapshot.exams || [];
        const lis = ['<li role="option" data-exam="auto" class="' +
            (!state.examIdLocked ? 'is-selected' : '') + '">' +
            '<span>Otomatik</span><span class="bx-em-count">' + (exams.length || 0) + ' sınav</span></li>'];
        for (const ex of exams) {
            const count = (snapshot.attempts || []).filter(a => a.exam_id === ex.exam_id).length;
            const sel = state.examIdLocked && ex.exam_id === examId;
            lis.push(
                '<li role="option" data-exam="' + escapeAttr(ex.exam_id) + '" class="' + (sel ? 'is-selected' : '') + '">' +
                '<span>' + escapeHtml(ex.name || ex.exam_id) + '</span>' +
                '<span class="bx-em-count">' + count + '</span></li>'
            );
        }
        menu.innerHTML = lis.join('');
    }

    function bindExamPicker() {
        const btn = $('bx-exam-picker');
        const menu = $('bx-exam-menu');
        if (!btn || !menu) return;

        const close = () => {
            menu.hidden = true;
            btn.setAttribute('aria-expanded', 'false');
        };
        const open = () => {
            menu.hidden = false;
            btn.setAttribute('aria-expanded', 'true');
        };

        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            menu.hidden ? open() : close();
        });
        menu.addEventListener('click', (e) => {
            const li = e.target.closest('li[data-exam]');
            if (!li) return;
            const value = li.dataset.exam;
            close();
            if (value === 'auto') {
                // Auto-mode: pickActiveExam runs during refresh
                setExam(null, { locked: false });
            } else {
                setExam(value, { locked: true });
            }
        });
        document.addEventListener('click', (e) => {
            if (menu.hidden) return;
            if (e.target === btn || btn.contains(e.target)) return;
            if (menu.contains(e.target)) return;
            close();
        });
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && !menu.hidden) close();
        });
    }

    // ── Reset Demo (Feature 3) ───────────────────────────────────────────
    function bindResetDemo() {
        const link = $('bx-reset-demo');
        if (!link) return;
        link.addEventListener('click', async (e) => {
            e.preventDefault();
            if (state.busy) return;
            const msg = (window.t ? window.t('confirm.resetDemo') : 'FaceVerify Campus demo verileri sıfırlansın mı?');
            if (!window.confirm(msg)) return;
            state.busy = true;
            try {
                await api.resetDemo();
                // Reset client-side state so the next refresh picks the seeded exam fresh
                state.selectedAttemptId = null;
                state.focusedStudentId = null;
                state.focusCleared = false;
                state.examId = null;
                state.examIdLocked = false;
                state.lastQueueIds = new Map();
                state.lastReviewIds = null;
                state.lastRendered = {};
                state.firstRenderDone = false;
                applySkeletons(true);
                await refresh();
                toast('Demo sıfırlandı.', 'success');
            } catch (err) {
                reportError('Sıfırlama başarısız', err);
            } finally {
                state.busy = false;
            }
        });
    }

    // ── tabs / row select / pager / shortcuts ────────────────────────────
    function bindTabs() {
        const tabs = $('bx-queue-tabs');
        if (!tabs) return;
        tabs.addEventListener('click', (e) => {
            const btn = e.target.closest('.bx-tab');
            if (!btn) return;
            const f = btn.dataset.filter;
            if (!f) return;
            setFilter(f);
        });
    }

    function bindRowSelect() {
        const tbody = $('bx-queue-tbody');
        if (!tbody) return;
        tbody.addEventListener('click', (e) => {
            const tr = e.target.closest('tr');
            if (!tr || !tr.dataset.attemptId) return;
            setFocus({ attemptId: tr.dataset.attemptId, source: 'user-queue' });
        });
    }

    function bindPager() {
        const pager = $('bx-queue-pager');
        if (!pager) return;
        pager.addEventListener('click', (e) => {
            const btn = e.target.closest('button');
            if (!btn || !btn.dataset.page) return;
            state.page = Math.max(1, parseInt(btn.dataset.page, 10));
            if (state.snapshot && state.examId) renderQueue(state.snapshot, state.examId);
        });
    }

    function bindGenericActive(selector) {
        const items = document.querySelectorAll(selector);
        items.forEach(it => {
            it.addEventListener('click', (e) => {
                if (it.matches('.bx-pager-arrow')) return;
                if (it.tagName === 'A') e.preventDefault();
                items.forEach(x => x.classList.remove('is-active'));
                it.classList.add('is-active');
            });
        });
    }

    function bindShortcuts() {
        document.addEventListener('keydown', (e) => {
            const tag = (e.target && e.target.tagName) ? e.target.tagName.toLowerCase() : '';
            if (tag === 'input' || tag === 'textarea') return;
            const k = (e.key || '').toLowerCase();
            if (k === 'a') reviewSelected('approve', 'Erişim onaylandı.');
            else if (k === 'd') reviewSelected('deny', 'Erişim reddedildi.');
            else if (k === 'm') reviewSelected('fallback', 'Yedek kontrol istendi.');
            else if (k === 'n') advanceNext();
        });
    }

    function bindVisibility() {
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) stopPolling();
            else { refresh(); startPolling(); }
        });
    }

    // ── boot ─────────────────────────────────────────────────────────────
    document.addEventListener('DOMContentLoaded', () => {
        tickClock();
        state.clockHandle = setInterval(tickClock, 1000);
        bindTabs();
        bindRowSelect();
        bindPager();
        bindGenericActive('.bx-sidebar .bx-nav-link');
        bindActions();
        bindResetDemo();
        bindExamPicker();
        bindAuditDrawer();
        bindSortMenu();
        bindQueueSearch();
        bindSoundToggle();
        bindReverify();
        bindSimulationToolbar();
        bindCloseDetail();
        bindAuditLink();
        bindHelpLink();
        bindSidebarNav();
        bindShortcuts();
        bindVisibility();

        // Hydrate persisted state before first refresh
        hydrateExamSelectionFromStorage();
        hydrateSortModeFromStorage();
        updateSortMenuSelection();

        // Skeleton placeholders until first refresh resolves
        applySkeletons(true);

        // First paint, then start polling
        refresh().finally(() => startPolling());
    });
})();
