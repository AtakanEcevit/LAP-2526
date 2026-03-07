/**
 * Biometric Verification UI — Application Logic
 *
 * Handles tab switching, drag-and-drop, API interactions,
 * and result rendering with animations.
 */

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initDropZones();
    initVerify();
    initCompare();
    initEnroll();
    checkHealth();
    refreshUserList();
});


/* ═══════════════════════════════════════════════════════════════════════
   Tab Navigation
   ═══════════════════════════════════════════════════════════════════════ */

function initTabs() {
    const btns = document.querySelectorAll('.tab-btn');
    btns.forEach(btn => {
        btn.addEventListener('click', () => {
            btns.forEach(b => { b.classList.remove('active'); b.setAttribute('aria-selected', 'false'); });
            btn.classList.add('active');
            btn.setAttribute('aria-selected', 'true');

            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');

            // Refresh users when switching to enroll or verify
            if (btn.dataset.tab === 'enroll' || btn.dataset.tab === 'verify') {
                refreshUserList();
            }
        });
    });
}


/* ═══════════════════════════════════════════════════════════════════════
   Drop Zone Setup
   ═══════════════════════════════════════════════════════════════════════ */

function initDropZones() {
    document.querySelectorAll('.drop-zone').forEach(zone => {
        const input = zone.querySelector('input[type="file"]');

        // Drag events
        zone.addEventListener('dragover', e => {
            e.preventDefault();
            zone.classList.add('drag-over');
        });
        zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));

        zone.addEventListener('drop', e => {
            e.preventDefault();
            zone.classList.remove('drag-over');
            if (e.dataTransfer.files.length > 0) {
                input.files = e.dataTransfer.files;
                input.dispatchEvent(new Event('change'));
            }
        });

        // File selected — show preview
        input.addEventListener('change', () => {
            if (input.files.length > 0) {
                zone.classList.add('has-file');
                updateDropZonePreview(zone, input.files);
            }
        });
    });
}

function updateDropZonePreview(zone, files) {
    // Remove old previews
    zone.querySelectorAll('.dz-preview, .dz-file-name').forEach(el => el.remove());

    const icon = zone.querySelector('.dz-icon');
    const text = zone.querySelector('.dz-text');

    if (files.length === 1) {
        if (icon) icon.style.display = 'none';
        if (text) text.textContent = files[0].name;

        // Image preview
        const reader = new FileReader();
        reader.onload = e => {
            const img = document.createElement('img');
            img.className = 'dz-preview';
            img.src = e.target.result;
            zone.appendChild(img);
        };
        reader.readAsDataURL(files[0]);
    } else {
        if (icon) icon.style.display = 'none';
        if (text) text.textContent = `${files.length} files selected`;
    }
}

function resetDropZone(zone) {
    const input = zone.querySelector('input[type="file"]');
    input.value = '';
    zone.classList.remove('has-file');
    zone.querySelectorAll('.dz-preview, .dz-file-name').forEach(el => el.remove());
    const icon = zone.querySelector('.dz-icon');
    const text = zone.querySelector('.dz-text');
    if (icon) icon.style.display = '';
    if (text) text.textContent = text.dataset.original || 'Drop image here or click to browse';
}


/* ═══════════════════════════════════════════════════════════════════════
   Health Check
   ═══════════════════════════════════════════════════════════════════════ */

async function checkHealth() {
    try {
        const data = await api.health();
        document.getElementById('status-text').textContent =
            `Online | ${data.models_available.length} models`;
    } catch {
        document.getElementById('status-text').textContent = 'Offline';
        document.querySelector('.status-dot').style.background = 'var(--accent-red)';
        document.querySelector('.status-badge').style.color = 'var(--accent-red)';
        document.querySelector('.status-badge').style.borderColor = 'rgba(239, 68, 68, 0.3)';
        document.querySelector('.status-badge').style.background = 'rgba(239, 68, 68, 0.08)';
    }
}


/* ═══════════════════════════════════════════════════════════════════════
   Verify Tab
   ═══════════════════════════════════════════════════════════════════════ */

function initVerify() {
    const userSelect = document.getElementById('verify-user');
    const fileInput = document.getElementById('verify-file');
    const btn = document.getElementById('verify-btn');

    const updateBtnState = () => {
        btn.disabled = !userSelect.value || !fileInput.files.length;
    };

    userSelect.addEventListener('change', () => {
        updateBtnState();
        // Show user info
        const opt = userSelect.selectedOptions[0];
        if (opt && opt.dataset.info) {
            document.getElementById('verify-user-info').textContent = opt.dataset.info;
        }
    });
    fileInput.addEventListener('change', updateBtnState);

    btn.addEventListener('click', async () => {
        const userId = userSelect.value;
        const image = fileInput.files[0];
        if (!userId || !image) return;

        setLoading(btn, 'verify-spinner', true);
        try {
            const result = await api.verify(userId, image);
            showResult('verify', result, userId);
        } catch (err) {
            showToast(err.message, 'error');
        } finally {
            setLoading(btn, 'verify-spinner', false);
        }
    });
}


/* ═══════════════════════════════════════════════════════════════════════
   Compare Tab
   ═══════════════════════════════════════════════════════════════════════ */

function initCompare() {
    const file1 = document.getElementById('compare-file1');
    const file2 = document.getElementById('compare-file2');
    const btn = document.getElementById('compare-btn');

    const updateBtnState = () => {
        btn.disabled = !file1.files.length || !file2.files.length;
    };

    file1.addEventListener('change', updateBtnState);
    file2.addEventListener('change', updateBtnState);

    btn.addEventListener('click', async () => {
        const modality = document.getElementById('compare-modality').value;
        const model = document.getElementById('compare-model').value;
        const image1 = file1.files[0];
        const image2 = file2.files[0];
        if (!image1 || !image2) return;

        setLoading(btn, 'compare-spinner', true);
        try {
            const result = await api.compare(modality, model, image1, image2);
            showResult('compare', result);
        } catch (err) {
            showToast(err.message, 'error');
        } finally {
            setLoading(btn, 'compare-spinner', false);
        }
    });
}


/* ═══════════════════════════════════════════════════════════════════════
   Enroll Tab
   ═══════════════════════════════════════════════════════════════════════ */

function initEnroll() {
    const userIdInput = document.getElementById('enroll-user-id');
    const filesInput = document.getElementById('enroll-files');
    const btn = document.getElementById('enroll-btn');

    const updateBtnState = () => {
        btn.disabled = !userIdInput.value.trim() || !filesInput.files.length;
    };

    userIdInput.addEventListener('input', updateBtnState);
    filesInput.addEventListener('change', updateBtnState);

    btn.addEventListener('click', async () => {
        const userId = userIdInput.value.trim();
        const modality = document.getElementById('enroll-modality').value;
        const model = document.getElementById('enroll-model').value;
        const files = filesInput.files;

        if (!userId || !files.length) return;

        setLoading(btn, 'enroll-spinner', true);
        try {
            const result = await api.enroll(userId, modality, model, files);
            showToast(
                `Enrolled "${result.user_id}" with ${result.sample_count} sample(s)`,
                'success'
            );
            // Reset form
            userIdInput.value = '';
            resetDropZone(document.getElementById('enroll-drop'));
            btn.disabled = true;
            refreshUserList();
        } catch (err) {
            showToast(err.message, 'error');
        } finally {
            setLoading(btn, 'enroll-spinner', false);
        }
    });
}


/* ═══════════════════════════════════════════════════════════════════════
   User List
   ═══════════════════════════════════════════════════════════════════════ */

async function refreshUserList() {
    try {
        const users = await api.listUsers();

        // Update Verify tab dropdown
        const select = document.getElementById('verify-user');
        const currentVal = select.value;
        select.innerHTML = '<option value="" disabled selected>Select a user...</option>';

        users.forEach(u => {
            const opt = document.createElement('option');
            opt.value = u.user_id;
            opt.textContent = u.user_id;
            opt.dataset.info = `${u.modality} | ${u.model_type} | ${u.sample_count} sample(s)`;
            select.appendChild(opt);
        });

        if (currentVal) select.value = currentVal;

        // Update Enroll tab user list
        const container = document.getElementById('user-list-container');
        if (users.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">&#128100;</div>
                    <div>No users enrolled yet</div>
                </div>`;
            return;
        }

        let html = `<table class="user-table">
            <thead><tr>
                <th>User ID</th>
                <th>Modality</th>
                <th>Model</th>
                <th>Samples</th>
                <th>Enrolled</th>
                <th></th>
            </tr></thead><tbody>`;

        users.forEach(u => {
            const time = u.enrolled_at ? u.enrolled_at.substring(0, 16).replace('T', ' ') : '';
            html += `<tr>
                <td class="user-id-cell">${escapeHtml(u.user_id)}</td>
                <td><span class="badge badge-modality">${u.modality}</span></td>
                <td><span class="badge badge-model">${u.model_type}</span></td>
                <td>${u.sample_count}</td>
                <td style="font-size:0.75rem;color:var(--text-muted)">${time}</td>
                <td><button class="btn btn-danger" onclick="deleteUser('${escapeHtml(u.user_id)}')">Delete</button></td>
            </tr>`;
        });

        html += '</tbody></table>';
        container.innerHTML = html;

    } catch (err) {
        console.error('Failed to refresh users:', err);
    }
}

async function deleteUser(userId) {
    try {
        await api.deleteUser(userId);
        showToast(`Deleted user "${userId}"`, 'info');
        refreshUserList();
    } catch (err) {
        showToast(err.message, 'error');
    }
}


/* ═══════════════════════════════════════════════════════════════════════
   Result Rendering
   ═══════════════════════════════════════════════════════════════════════ */

function showResult(prefix, result, userId) {
    const panel = document.getElementById(`${prefix}-result`);
    panel.classList.add('visible');

    // Show validation warnings if present
    renderValidationWarnings(prefix, result.validation);

    // Verdict
    const isMatch = result.match;
    document.getElementById(`${prefix}-verdict-icon`).textContent = isMatch ? '&#x2705;' : '&#x274C;';
    document.getElementById(`${prefix}-verdict-icon`).innerHTML = isMatch ? '&#x2705;' : '&#x274C;';

    const label = document.getElementById(`${prefix}-verdict-label`);
    label.textContent = isMatch ? 'MATCH' : 'NO MATCH';
    label.className = `verdict-label ${isMatch ? 'match' : 'no-match'}`;

    // Score bar — animate after a tick
    const fill = document.getElementById(`${prefix}-score-fill`);
    fill.style.width = '0%';
    fill.className = `score-bar-fill ${isMatch ? 'match' : 'no-match'}`;
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            fill.style.width = `${Math.min(result.score * 100, 100)}%`;
        });
    });

    // Threshold marker
    const marker = document.getElementById(`${prefix}-threshold-marker`);
    marker.style.left = `${result.threshold * 100}%`;

    // Score label
    document.getElementById(`${prefix}-score-label`).textContent =
        `Score: ${result.score.toFixed(4)}`;

    // Stats
    document.getElementById(`${prefix}-stat-score`).textContent = result.score.toFixed(4);
    document.getElementById(`${prefix}-stat-threshold`).textContent = result.threshold.toFixed(4);

    if (prefix === 'verify') {
        document.getElementById('verify-stat-user').textContent = userId || '-';
    } else {
        document.getElementById('compare-stat-verdict').textContent = isMatch ? 'Match' : 'No Match';
    }

    // Scroll into view
    panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}


function renderValidationWarnings(prefix, validation) {
    const panel = document.getElementById(`${prefix}-result`);
    // Remove old warnings
    panel.querySelectorAll('.validation-warning').forEach(el => el.remove());

    if (!validation) return;

    const warnings = collectValidationWarnings(validation);
    if (warnings.length === 0) return;

    const banner = document.createElement('div');
    banner.className = 'validation-warning';

    let listHtml = warnings.map(w => `<li>${escapeHtml(w)}</li>`).join('');
    banner.innerHTML = `
        <span class="vw-icon">&#x26A0;&#xFE0F;</span>
        <div class="vw-text">
            <strong>Input Quality Warning</strong>
            <ul>${listHtml}</ul>
        </div>`;

    // Insert at the top of the panel
    panel.prepend(banner);
}


function collectValidationWarnings(validation) {
    const warnings = [];
    if (!validation) return warnings;

    if (typeof validation === 'object' && !Array.isArray(validation)) {
        // Could be {image1: {...}, image2: {...}} or {passed, warnings, confidence}
        if (validation.warnings && Array.isArray(validation.warnings)) {
            warnings.push(...validation.warnings);
        } else {
            for (const [key, val] of Object.entries(validation)) {
                if (val && typeof val === 'object' && val.warnings) {
                    val.warnings.forEach(w => {
                        warnings.push(`${key}: ${w}`);
                    });
                }
            }
        }
    }
    return warnings;
}


/* ═══════════════════════════════════════════════════════════════════════
   Utilities
   ═══════════════════════════════════════════════════════════════════════ */

function setLoading(btn, spinnerId, loading) {
    const spinner = document.getElementById(spinnerId);
    if (loading) {
        btn.disabled = true;
        spinner.style.display = 'inline-block';
    } else {
        btn.disabled = false;
        spinner.style.display = 'none';
    }
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(40px)';
        toast.style.transition = 'all 300ms ease-in';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}
