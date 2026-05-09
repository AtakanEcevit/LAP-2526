const API_BASE = window.location.origin;

async function parseResponse(resp, fallback) {
    if (resp.ok) {
        const text = await resp.text();
        return text ? JSON.parse(text) : {};
    }
    const err = await resp.json().catch(() => ({ detail: fallback }));
    throw new Error(err.detail || fallback);
}

const api = {
    async health() {
        const resp = await fetch(`${API_BASE}/campus/status`);
        return parseResponse(resp, 'Health check failed');
    },

    async snapshot() {
        const resp = await fetch(`${API_BASE}/campus`);
        return parseResponse(resp, 'Failed to load campus data');
    },

    async users() {
        const resp = await fetch(`${API_BASE}/users`);
        return parseResponse(resp, 'Failed to load enrollment data');
    },

    async resetDemo() {
        const resp = await fetch(`${API_BASE}/campus/reset`, { method: 'POST' });
        return parseResponse(resp, 'Failed to reset demo');
    },

    async saveCourse(course) {
        const form = new FormData();
        Object.entries(course).forEach(([key, value]) => form.append(key, value));
        const resp = await fetch(`${API_BASE}/campus/courses`, {
            method: 'POST',
            body: form
        });
        return parseResponse(resp, 'Failed to save course');
    },

    async saveExam(exam) {
        const form = new FormData();
        Object.entries(exam).forEach(([key, value]) => form.append(key, value));
        const resp = await fetch(`${API_BASE}/campus/exams`, {
            method: 'POST',
            body: form
        });
        return parseResponse(resp, 'Failed to save exam');
    },

    async importRoster(courseId, file) {
        const form = new FormData();
        form.append('course_id', courseId);
        form.append('roster', file);
        const resp = await fetch(`${API_BASE}/campus/roster/import`, {
            method: 'POST',
            body: form
        });
        return parseResponse(resp, 'Failed to import roster');
    },

    async examRoster(examId) {
        const resp = await fetch(`${API_BASE}/campus/exams/${encodeURIComponent(examId)}/roster`);
        return parseResponse(resp, 'Failed to load exam roster');
    },

    async getAttempt(attemptId) {
        const resp = await fetch(`${API_BASE}/campus/attempts/${encodeURIComponent(attemptId)}`);
        return parseResponse(resp, 'Failed to load attempt');
    },

    async listAttempts(examId, studentId) {
        const params = new URLSearchParams();
        if (examId) params.set('exam_id', examId);
        if (studentId) params.set('student_id', studentId);
        const resp = await fetch(`${API_BASE}/campus/attempts?${params.toString()}`);
        return parseResponse(resp, 'Failed to load attempt history');
    },

    async enrollStudent(studentId, modelType, files) {
        const form = new FormData();
        form.append('model_type', modelType);
        Array.from(files).forEach(file => form.append('images', file));
        const resp = await fetch(`${API_BASE}/campus/students/${encodeURIComponent(studentId)}/enroll`, {
            method: 'POST',
            body: form
        });
        return parseResponse(resp, 'Failed to enroll student');
    },

    async verifyStudent(examId, studentId, file) {
        const form = new FormData();
        form.append('student_id', studentId);
        form.append('image', file);
        const resp = await fetch(`${API_BASE}/campus/exams/${encodeURIComponent(examId)}/verify`, {
            method: 'POST',
            body: form
        });
        return parseResponse(resp, 'Verification failed');
    },

    async reviewAttempt(attemptId, reviewer, action, reason) {
        const form = new FormData();
        form.append('reviewer', reviewer);
        form.append('action', action);
        form.append('reason', reason);
        const resp = await fetch(`${API_BASE}/campus/attempts/${encodeURIComponent(attemptId)}/review`, {
            method: 'POST',
            body: form
        });
        return parseResponse(resp, 'Failed to review attempt');
    },

    async audit() {
        const resp = await fetch(`${API_BASE}/campus/audit`);
        return parseResponse(resp, 'Failed to load audit log');
    },

    async modelLabCompare(modelType, imageA, imageB) {
        const form = new FormData();
        form.append('model_type', modelType);
        form.append('image1', imageA);
        form.append('image2', imageB);
        const resp = await fetch(`${API_BASE}/campus/model-lab/compare`, {
            method: 'POST',
            body: form
        });
        return parseResponse(resp, 'Model comparison failed');
    }
};
