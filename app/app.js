const state = { data: null, selected: null };
const budget = document.querySelector("#budget");
const model = document.querySelector("#model");

function pct(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function money(order) {
  return new Intl.NumberFormat("en-US", { style: "currency", currency: order.amount.currency_code }).format(order.amount.amount_minor / 100);
}

function ranked(name) {
  return [...state.data.payments].sort((a, b) => b.scores[name] - a.scores[name]);
}

function queueStats(name, share) {
  const rows = ranked(name);
  const alerts = Math.max(1, Math.ceil(rows.length * share));
  const caught = rows.slice(0, alerts).filter((row) => row.label === 1).length;
  const frauds = rows.filter((row) => row.label === 1).length;
  return { alerts, caught, recall: caught / frauds, precision: caught / alerts };
}

function renderCards() {
  const share = Number(budget.value) / 100;
  document.querySelector("#budgetLabel").textContent = `${budget.value}%`;
  document.querySelector("#cards").innerHTML = Object.keys(state.data.payments[0].scores).map((name) => {
    const stats = queueStats(name, share);
    return `<article class="card ${name === model.value ? "active" : ""}">
      <h3>${name}</h3><div class="metric">${pct(stats.recall)}</div>
      <p>${stats.caught} scams in ${stats.alerts} reviews. Precision ${pct(stats.precision)}.</p>
    </article>`;
  }).join("");
}

function renderQueue() {
  const share = Number(budget.value) / 100;
  const rows = ranked(model.value).slice(0, Math.max(1, Math.ceil(state.data.payments.length * share)));
  if (!state.selected || !rows.includes(state.selected)) state.selected = rows[0];
  document.querySelector("#queue").innerHTML = rows.map((row, index) => {
    const order = row.canonical.payment_order;
    return `<tr data-id="${row.payment_id}" class="${row === state.selected ? "selected" : ""}">
      <td>${index + 1}</td><td>${order.payment_order_id}</td><td>${order.rail.replace("PAYMENT_RAIL_", "")}</td>
      <td>${money(order)}</td><td>${row.scores[model.value].toFixed(3)}</td>
      <td>${row.evidence.map((item) => `<span class="chip">${item}</span>`).join("") || "No listed flags"}</td>
      <td class="${row.label ? "fraud" : ""}">${row.label ? row.typology.replaceAll("_", " ") : "legitimate"}</td>
    </tr>`;
  }).join("");
  document.querySelectorAll("#queue tr").forEach((tr) => tr.addEventListener("click", () => {
    state.selected = state.data.payments.find((row) => row.payment_id === tr.dataset.id);
    renderQueue();
  }));
  renderDetails();
}

function renderDetails() {
  const row = state.selected;
  document.querySelector("#payload").textContent = JSON.stringify(row.canonical, null, 2);
  const action = row.hold_recommendation ? {
    fraud_evaluation: row.fraud_evaluation,
    hold_recommendation: row.hold_recommendation,
    customer_case: row.customer_case,
  } : { fraud_evaluation: row.fraud_evaluation, action: "Outside the precomputed TabPFN top 5%." };
  document.querySelector("#action").textContent = JSON.stringify(action, null, 2);
}

function render() {
  renderCards();
  renderQueue();
}

fetch("../results/demo.json")
  .then((response) => response.json())
  .then((data) => {
    state.data = data;
    model.innerHTML = Object.keys(data.payments[0].scores).map((name) => `<option>${name}</option>`).join("");
    render();
  });

budget.addEventListener("input", render);
model.addEventListener("change", () => { state.selected = null; render(); });
