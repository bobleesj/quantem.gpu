// Keep the mobile hamburger working across Sphinx theme version drift.
//
// sphinx-book-theme and pydata-sphinx-theme both wire the sidebar drawer to
// document.querySelector(".primary-toggle") -- the first match. Newer pydata
// themes render a hidden header button before the visible book-theme button,
// so the click handler can land on the hidden control. Forward visible button
// clicks to the wired control. This is intentionally idempotent.
document.addEventListener("DOMContentLoaded", () => {
  const toggles = Array.from(document.querySelectorAll(".primary-toggle"));
  if (toggles.length < 2) return;

  const wired = toggles[0];
  for (const button of toggles.slice(1)) {
    if (button.dataset.navToggleFixed) continue;
    button.dataset.navToggleFixed = "1";
    button.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      wired.click();
    });
  }
});
