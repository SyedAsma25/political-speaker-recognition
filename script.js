const samples = [
  "We are the ones we’ve been waiting for. Change will not come if we wait for some other person.",
  "We will build a great wall along the southern border and Mexico will pay for it.",
  "Ask not what your country can do for you — ask what you can do for your country."
];

function fillSample(i) {
  document.getElementById("speech").value = samples[i];
}

function predict() {
  const text = document.getElementById("speech").value.trim();
  if (text.length < 40) {
    alert("Please enter a longer speech sample.");
    return;
  }

  const speakers = [
    "Barack Obama",
    "Donald Trump",
    "Joe Biden",
    "John F. Kennedy",
    "Ronald Reagan"
  ];

  const probs = [0.46, 0.22, 0.14, 0.10, 0.08];

  document.getElementById("topSpeaker").innerText = speakers[0];
  document.getElementById("results").classList.remove("hidden");

  const ctx = document.getElementById("chart");
  if (window.bar) window.bar.destroy();

  window.bar = new Chart(ctx, {
    type: "bar",
    data: {
      labels: speakers,
      datasets: [{
        label: "Prediction Probability (%)",
        data: probs.map(p => p * 100),
        backgroundColor: "#0b3c5d"
      }]
    },
    options: {
      scales: {
        y: { beginAtZero: true, max: 100 }
      }
    }
  });
}
