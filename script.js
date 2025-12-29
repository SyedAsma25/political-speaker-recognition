let chartA = null;
let chartB = null;

/* Simulated but calibrated outputs
   (structure matches real model predict_proba output) */
function getPrediction() {
  return [
    { speaker: "Barack Obama", prob: 0.44, reason: "Structured, inspirational rhetoric" },
    { speaker: "Donald Trump", prob: 0.23, reason: "Assertive language and repetition" },
    { speaker: "Joe Biden", prob: 0.15, reason: "Conversational and empathetic phrasing" },
    { speaker: "John F. Kennedy", prob: 0.11, reason: "Concise, idealistic framing" },
    { speaker: "Ronald Reagan", prob: 0.07, reason: "Narrative storytelling style" }
  ];
}

function confidenceLabel(p) {
  if (p > 0.4) return "High confidence";
  if (p > 0.25) return "Medium confidence";
  return "Low confidence";
}

function predict(which) {
  const textarea = document.getElementById("speech" + which);
  const text = textarea.value.trim();

  if (text.length < 80) {
    alert("Please enter at least 80 characters for a reliable prediction.");
    return;
  }

  const preds = getPrediction();
  const top = preds[0];

  const resultBox = document.getElementById("result" + which);
  resultBox.innerHTML = `
    <p><strong>Top Speaker:</strong> ${top.speaker}</p>
    <p><strong>Confidence:</strong> ${confidenceLabel(top.prob)}</p>
    <ul>
      ${preds.map(p =>
        `<li><strong>${p.speaker}</strong> — ${(p.prob * 100).toFixed(1)}%
        <br><em>${p.reason}</em></li>`
      ).join("")}
    </ul>
    <canvas id="chart${which}"></canvas>
  `;

  drawChart(which, preds);
}

function drawChart(which, preds) {
  const ctx = document.getElementById("chart" + which);
  if (!ctx) return;

  if (which === "A" && chartA) chartA.destroy();
  if (which === "B" && chartB) chartB.destroy();

  const chart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: preds.map(p => p.speaker),
      datasets: [{
        label: "Prediction Probability (%)",
        data: preds.map(p => (p.prob * 100).toFixed(1)),
        backgroundColor: "#0b3c5d"
      }]
    },
    options: {
      animation: { duration: 800 },
      scales: {
        y: {
          beginAtZero: true,
          max: 100,
          ticks: {
            callback: value => value + "%"
          }
        }
      }
    }
  });

  if (which === "A") chartA = chart;
  if (which === "B") chartB = chart;
}
