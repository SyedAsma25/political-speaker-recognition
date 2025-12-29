let predictions = [];

fetch("./predictions.json")
  .then(res => res.json())
  .then(data => {
    predictions = data;
    console.log("Loaded real predictions:", predictions.length);
  });

function predict(which) {
  if (predictions.length === 0) {
    alert("Predictions not loaded yet.");
    return;
  }

  const idx = Math.floor(Math.random() * predictions.length);
  const p = predictions[idx];

  const resultBox = document.getElementById("result" + which);

  resultBox.innerHTML = `
    <p><strong>Top Prediction:</strong> ${p.top_prediction}</p>
    <ul>
      ${p.top_3.map(
        r => `<li>${r.speaker} — ${(r.probability * 100).toFixed(1)}%</li>`
      ).join("")}
    </ul>
    <p class="note">
      Predictions are based on linguistic style, not political ideology.
    </p>
  `;
}
