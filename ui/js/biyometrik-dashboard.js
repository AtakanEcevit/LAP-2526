(() => {
    const TR_DAYS = ['Pazar', 'Pazartesi', 'Salı', 'Çarşamba', 'Perşembe', 'Cuma', 'Cumartesi'];
    const TR_MONTHS = ['Ocak', 'Şubat', 'Mart', 'Nisan', 'Mayıs', 'Haziran',
                       'Temmuz', 'Ağustos', 'Eylül', 'Ekim', 'Kasım', 'Aralık'];

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

    function tickClock() {
        const now = new Date();
        const time = document.getElementById('bx-clock-time');
        const date = document.getElementById('bx-clock-date');
        const upd = document.getElementById('bx-last-update');
        if (time) time.textContent = fmtClock(now);
        if (date) date.textContent = fmtDate(now);
        if (upd) upd.textContent = fmtClockSec(now);
    }

    // Queue tab switch (visual only)
    function bindTabs() {
        const tabs = document.querySelectorAll('.bx-tabs .bx-tab');
        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                tabs.forEach(t => t.classList.remove('is-active'));
                tab.classList.add('is-active');
            });
        });
    }

    // Queue row selection
    function bindRowSelect() {
        const rows = document.querySelectorAll('.bx-queue tbody tr');
        rows.forEach(row => {
            row.addEventListener('click', (e) => {
                if (e.target.closest('button')) return;
                rows.forEach(r => r.classList.remove('is-selected'));
                row.classList.add('is-selected');
            });
        });
    }

    // Pager + sidebar nav (visual only)
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

    // Keyboard shortcuts: A onayla, D reddet, N sonraki, M daha sonra
    function bindShortcuts() {
        const flash = (sel) => {
            const btn = document.querySelector(sel);
            if (!btn) return;
            btn.style.transform = 'scale(0.96)';
            setTimeout(() => { btn.style.transform = ''; }, 150);
        };
        document.addEventListener('keydown', (e) => {
            if (e.target.matches('input, textarea')) return;
            const k = e.key.toLowerCase();
            if (k === 'a') flash('.bx-btn-success');
            else if (k === 'd') flash('.bx-btn-danger');
            else if (k === 'n') flash('.bx-btn-link');
            else if (k === 'm') flash('.bx-btn-outline');
        });
    }

    document.addEventListener('DOMContentLoaded', () => {
        tickClock();
        setInterval(tickClock, 1000);
        bindTabs();
        bindRowSelect();
        bindGenericActive('.bx-sidebar .bx-nav-link');
        bindGenericActive('.bx-pager button');
        bindShortcuts();
    });
})();
