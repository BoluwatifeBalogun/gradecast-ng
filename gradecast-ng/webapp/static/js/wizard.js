/* Five-step prediction form. Works as one long form without JavaScript. */
(function () {
  'use strict';
  const form = document.getElementById('wizard');
  if (!form) return;
  const panels = Array.from(form.querySelectorAll('.panel'));
  const tabs = Array.from(form.querySelectorAll('.progress button'));
  const now = document.getElementById('step-now');
  let current = -1;
  let furthest = 0;

  panels.forEach((p, i) => {
    if (p.querySelector('.has-error')) tabs[i].classList.add('has-error');
  });

  function show(index, back) {
    index = Math.max(0, Math.min(panels.length - 1, index));
    furthest = Math.max(furthest, index);
    panels.forEach((p, i) => {
      p.classList.toggle('is-current', i === index);
      p.classList.toggle('back', !!back);
    });
    tabs.forEach((t, i) => {
      t.classList.toggle('is-current', i === index);
      t.classList.toggle('is-done', i < index);
      t.setAttribute('aria-selected', String(i === index));
      t.disabled = i > furthest && !form.querySelector('.has-error');
    });
    now.textContent = index + 1;
    if (current !== -1) {
      const heading = panels[index].querySelector('h2');
      heading.setAttribute('tabindex', '-1');
      heading.focus({ preventScroll: true });
      form.scrollIntoView({ behavior: GC.reduce ? 'auto' : 'smooth', block: 'start' });
    }
    current = index;
  }

  /* Step one has the only free-text required field: check it in place */
  function stepValid(index) {
    if (index !== 0) return true;
    const name = document.getElementById('student_name');
    const field = name.closest('.field');
    const old = field.querySelector('.error-text');
    if (old) old.remove();
    if (name.value.trim().length >= 2) { field.classList.remove('has-error'); name.removeAttribute('aria-invalid'); return true; }
    field.classList.add('has-error');
    name.setAttribute('aria-invalid', 'true');
    const msg = document.createElement('span');
    msg.className = 'error-text'; msg.id = 'err-student_name';
    msg.textContent = "Enter the student's name.";
    field.appendChild(msg);
    name.setAttribute('aria-describedby', 'err-student_name');
    name.focus();
    return false;
  }

  form.addEventListener('click', (e) => {
    const next = e.target.closest('[data-next]');
    const back = e.target.closest('[data-back]');
    const go = e.target.closest('[data-go]');
    if (next && stepValid(current)) show(current + 1);
    if (back) show(current - 1, true);
    if (go && !go.disabled) {
      const target = Number(go.dataset.go);
      if (target < current || stepValid(0)) show(target, target < current);
    }
  });

  /* Enter in a text field moves on instead of submitting half a form */
  form.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.target.matches('input.input') && current < panels.length - 1) {
      e.preventDefault();
      if (stepValid(current)) show(current + 1);
    }
  });
  form.addEventListener('submit', (e) => {
    if (!stepValid(0)) { e.preventDefault(); show(0, true); }
  });

  if (form.querySelector('.has-error')) furthest = panels.length - 1;
  show(Number(form.dataset.start) || 0);
})();
