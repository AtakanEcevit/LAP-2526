/**
 * API wrapper for the Biometric Verification backend.
 * All functions return Promises with parsed JSON responses.
 */

const API_BASE = 'http://127.0.0.1:8000';  // Backend API server

const api = {

    /**
     * Health check
     * @returns {Promise<{status, models_available, models_loaded, enrollment_count}>}
     */
    async health() {
        const resp = await fetch(`${API_BASE}/health`);
        if (!resp.ok) throw new Error(`Health check failed: ${resp.status}`);
        return resp.json();
    },

    /**
     * Compare two images directly (no enrollment needed)
     * @param {string} modality - "signature", "face", or "fingerprint"
     * @param {string} model - "siamese" or "prototypical"
     * @param {File} image1
     * @param {File} image2
     * @returns {Promise<{match, score, threshold}>}
     */
    async compare(modality, model, image1, image2) {
        const form = new FormData();
        form.append('modality', modality);
        form.append('model', model);
        form.append('image1', image1);
        form.append('image2', image2);

        const resp = await fetch(`${API_BASE}/compare`, { method: 'POST', body: form });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(err.detail || 'Comparison failed');
        }
        return resp.json();
    },

    /**
     * Enroll a user with reference images
     * @param {string} userId
     * @param {string} modality
     * @param {string} model
     * @param {FileList|File[]} images
     * @returns {Promise<{user_id, sample_count, message}>}
     */
    async enroll(userId, modality, model, images) {
        const form = new FormData();
        form.append('user_id', userId);
        form.append('modality', modality);
        form.append('model', model);
        for (const img of images) {
            form.append('images', img);
        }

        const resp = await fetch(`${API_BASE}/enroll`, { method: 'POST', body: form });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(err.detail || 'Enrollment failed');
        }
        return resp.json();
    },

    /**
     * Verify a query image against an enrolled user
     * @param {string} userId
     * @param {File} image
     * @returns {Promise<{user_id, match, score, threshold}>}
     */
    async verify(userId, image) {
        const form = new FormData();
        form.append('user_id', userId);
        form.append('image', image);

        const resp = await fetch(`${API_BASE}/verify`, { method: 'POST', body: form });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(err.detail || 'Verification failed');
        }
        return resp.json();
    },

    /**
     * List all enrolled users
     * @returns {Promise<Array<{user_id, modality, model_type, sample_count, enrolled_at}>>}
     */
    async listUsers() {
        const resp = await fetch(`${API_BASE}/users`);
        if (!resp.ok) throw new Error('Failed to fetch users');
        return resp.json();
    },

    /**
     * Delete an enrolled user
     * @param {string} userId
     * @returns {Promise<{deleted, user_id}>}
     */
    async deleteUser(userId) {
        const resp = await fetch(`${API_BASE}/users/${encodeURIComponent(userId)}`, {
            method: 'DELETE'
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            throw new Error(err.detail || 'Delete failed');
        }
        return resp.json();
    }
};
