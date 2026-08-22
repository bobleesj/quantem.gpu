// Add lightweight, dependency-free filters to the platform matrices.
//
// The Markdown tables remain the canonical, printable evidence. This script
// only filters rendered rows; it never rewrites values or fetches another data
// source. Keeping that boundary makes the static Pages build reproducible.
document.addEventListener("DOMContentLoaded", () => {
  const normalize = (value) => value.replace(/\s+/g, " ").trim();

  for (const table of document.querySelectorAll("table")) {
    const headerRow = table.tHead?.rows[0];
    const body = table.tBodies[0];
    if (!headerRow || !body || body.rows.length < 4) continue;

    const headers = Array.from(headerRow.cells, (cell) => normalize(cell.textContent));
    const platformIndex = headers.indexOf("Platform");
    if (platformIndex < 0 || table.dataset.qgpuFilterReady) continue;
    table.dataset.qgpuFilterReady = "1";

    const tools = document.createElement("div");
    tools.className = "qgpu-table-tools";
    tools.setAttribute("role", "search");
    tools.setAttribute("aria-label", "Filter benchmark table");

    const search = document.createElement("input");
    search.type = "search";
    search.placeholder = "Filter rows";
    search.setAttribute("aria-label", "Filter table rows by text");
    tools.append(search);

    const selectors = [];
    for (const columnName of [
      "Platform",
      "Device tested",
      "Detector bin",
      "Cache/process state",
      "Gate",
    ]) {
      const columnIndex = headers.indexOf(columnName);
      if (columnIndex < 0) continue;

      const values = Array.from(
        new Set(
          Array.from(body.rows, (row) => normalize(row.cells[columnIndex]?.textContent || ""))
            .filter((value) => value && value !== "—")
        )
      ).sort((left, right) => left.localeCompare(right, undefined, { numeric: true }));
      if (values.length < 2 || values.length > 30) continue;

      const select = document.createElement("select");
      select.setAttribute("aria-label", `Filter by ${columnName}`);
      select.append(new Option(`All ${columnName.toLowerCase()}`, ""));
      for (const value of values) select.append(new Option(value, value));
      tools.append(select);
      selectors.push({ columnIndex, select });
    }

    const count = document.createElement("span");
    count.className = "qgpu-table-count";
    count.setAttribute("aria-live", "polite");
    tools.append(count);

    const applyFilters = () => {
      const query = normalize(search.value).toLocaleLowerCase();
      let visible = 0;
      for (const row of body.rows) {
        const textMatches = !query || normalize(row.textContent).toLocaleLowerCase().includes(query);
        const selectionsMatch = selectors.every(({ columnIndex, select }) => {
          const cellValue = normalize(row.cells[columnIndex]?.textContent || "");
          return !select.value || cellValue === select.value;
        });
        row.hidden = !(textMatches && selectionsMatch);
        if (!row.hidden) visible += 1;
      }
      count.textContent = `${visible} of ${body.rows.length} rows`;
    };

    search.addEventListener("input", applyFilters);
    for (const { select } of selectors) select.addEventListener("change", applyFilters);

    const container = table.closest(".pst-scrollable-table-container") || table;
    container.parentNode?.insertBefore(tools, container);
    applyFilters();
  }
});
