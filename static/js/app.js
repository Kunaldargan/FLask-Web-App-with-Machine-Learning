document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.flash').forEach((flash) => {
    window.setTimeout(() => {
      flash.style.opacity = '0';
      flash.style.transition = 'opacity .35s ease';
      window.setTimeout(() => flash.remove(), 400);
    }, 6000);
  });
});
