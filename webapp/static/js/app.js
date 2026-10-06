/* GradeCast NG: shared behaviour (no framework, no build step). */
(function () {
  'use strict';
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

  /* Scroll reveal */
  const revealables = $$('.reveal');
  if (reduce || !('IntersectionObserver' in window)) {
    revealables.forEach((el) => el.classList.add('is-in'));
  } else {
    const io = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add('is-in');
        io.unobserve(entry.target);
      });
    }, { threshold: 0.12, rootMargin: '0px 0px -6% 0px' });
    revealables.forEach((el) => io.observe(el));
  }

  /* Number ticker: counts up when first seen */
  function tick(el) {
    const target = parseFloat(el.dataset.count);
    const decimals = (el.dataset.count.split('.')[1] || '').length;
    const suffix = el.dataset.suffix || '';
    const fmt = (v) => v.toLocaleString('en-NG', { minimumFractionDigits: decimals, maximumFractionDigits: decimals }) + suffix;
    if (reduce || !isFinite(target)) { el.textContent = fmt(target); return; }
    const start = performance.now(), dur = 1400;
    (function frame(now) {
      const t = Math.max(0, Math.min(1, (now - start) / dur));
      el.textContent = fmt(target * (1 - Math.pow(1 - t, 4)));
      if (t < 1) requestAnimationFrame(frame);
    })(start);
  }
  if ('IntersectionObserver' in window) {
    const co = new IntersectionObserver((entries) => {
      entries.forEach((e) => { if (e.isIntersecting) { tick(e.target); co.unobserve(e.target); } });
    }, { threshold: 0.5 });
    $$('[data-count]').forEach((el) => co.observe(el));
  }

  /* Spotlight cards: glow follows the pointer */
  $$('.spot').forEach((card) => {
    card.addEventListener('pointermove', (e) => {
      const r = card.getBoundingClientRect();
      card.style.setProperty('--mx', (e.clientX - r.left) + 'px');
      card.style.setProperty('--my', (e.clientY - r.top) + 'px');
    });
  });

  /* Range inputs: filled track and live value */
  function paint(input) {
    const min = +input.min || 0, max = +input.max || 100;
    input.style.setProperty('--fill', ((input.value - min) / (max - min)) * 100 + '%');
    const out = input.id && document.querySelector('output[for="' + input.id + '"]');
    if (out) out.textContent = input.value + (input.dataset.unit || '');
  }
  $$('input[type="range"]').forEach((input) => {
    paint(input);
    input.addEventListener('input', () => paint(input));
  });
  window.GC = { paint: paint, reduce: reduce };

  /* Toasts */
  function dismiss(toast) {
    toast.classList.add('is-leaving');
    setTimeout(() => toast.remove(), 260);
  }
  $$('.toast').forEach((toast) => {
    toast.querySelector('[data-dismiss]').addEventListener('click', () => dismiss(toast));
    if (!toast.classList.contains('error')) setTimeout(() => dismiss(toast), 5000);
  });

  /* Mobile drawer */
  const sidebar = document.getElementById('sidebar');
  const scrim = document.getElementById('scrim');
  const menuBtn = document.getElementById('menu-btn');
  if (sidebar && menuBtn) {
    const setOpen = (open) => {
      sidebar.classList.toggle('is-open', open);
      scrim.classList.toggle('is-open', open);
      menuBtn.setAttribute('aria-expanded', String(open));
      if (open) { const first = sidebar.querySelector('a'); if (first) first.focus(); }
    };
    menuBtn.addEventListener('click', () => setOpen(!sidebar.classList.contains('is-open')));
    scrim.addEventListener('click', () => setOpen(false));
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') setOpen(false); });
  }

  /* Confirm before destructive actions; show a spinner on submit */
  document.addEventListener('submit', (e) => {
    const form = e.target;
    if (form.dataset.confirm && !window.confirm(form.dataset.confirm)) { e.preventDefault(); return; }
    const btn = form.querySelector('button[type="submit"][data-loading]');
    if (btn && !e.defaultPrevented) btn.classList.add('is-loading');
  });

  /* Sticky landing nav shadow */
  const nav = document.querySelector('.nav');
  if (nav) {
    const onScroll = () => nav.classList.toggle('is-stuck', window.scrollY > 24);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
  }

  /* SMOTE bars animate when their tile is revealed */
  $$('[data-reveal-bars]').forEach((bars) => {
    if (!('IntersectionObserver' in window) || reduce) { bars.classList.add('is-in'); return; }
    const bo = new IntersectionObserver((entries) => {
      if (entries[0].isIntersecting) { setTimeout(() => bars.classList.add('is-in'), 500); bo.disconnect(); }
    }, { threshold: 0.6 });
    bo.observe(bars);
  });

  /* Shared Chart.js look */
  if (window.Chart) {
    const C = window.Chart;
    C.defaults.font.family = '"Instrument Sans", "Segoe UI", system-ui, sans-serif';
    C.defaults.font.size = 13;
    C.defaults.color = '#53665d';
    C.defaults.plugins.legend.labels.usePointStyle = true;
    C.defaults.plugins.legend.labels.boxHeight = 8;
    C.defaults.plugins.tooltip.backgroundColor = '#062e22';
    C.defaults.plugins.tooltip.padding = 12;
    C.defaults.plugins.tooltip.cornerRadius = 10;
    C.defaults.plugins.tooltip.titleFont = { family: '"Bricolage Grotesque", sans-serif', size: 14, weight: '600' };
    C.defaults.maintainAspectRatio = false;
    if (reduce) C.defaults.animation = false;
    window.GC.classColors = ['#d24b35', '#e3a322', '#8fd3ad', '#1ea76c', '#00693e'];
    window.GC.grid = { color: '#e6eee9', drawTicks: false };
  }
})();
