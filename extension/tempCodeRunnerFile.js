// popup.js
document.getElementById('analyzeBtn').addEventListener('click', () => {
    const resultDiv = document.getElementById('analysisResult');
    resultDiv.innerHTML = '<div class="loader"></div><div style="text-align:center">Analyzing page content...</div>';
    
    // Send message to background script to get page text and analyze
    chrome.runtime.sendMessage({ action: "analyzePage" }, (response) => {
        if (response && response.error) {
            resultDiv.innerHTML = `<div style="color:red;">Error: ${response.error}</div>`;
        } else if (response) {
            displayResult(response);
        } else {
            resultDiv.innerHTML = `<div style="color:red;">No response from server. Make sure Flask is running.</div>`;
        }
    });
});

document.getElementById('openWebApp').addEventListener('click', () => {
    chrome.tabs.create({ url: 'http://127.0.0.1:5000' });
});

function displayResult(data) {
    const isFake = data.result === 'FAKE';
    const confidence = data.confidence;
    const fillColor = isFake ? '#dc2626' : '#16a34a';
    const resultDiv = document.getElementById('analysisResult');
    resultDiv.innerHTML = `
        <div class="result ${isFake ? 'fake' : 'real'}">
            ${isFake ? '⚠️ FAKE NEWS DETECTED' : '✅ REAL NEWS'}
        </div>
        <div class="confidence-bar">
            <div class="confidence-fill ${isFake ? 'fake-fill' : 'real-fill'}" style="width: ${confidence}%;"></div>
        </div>
        <div>Confidence: ${confidence}%</div>
        ${data.title ? `<div style="font-size:12px; margin-top:8px;">📰 ${data.title.substring(0, 60)}</div>` : ''}
        <div class="note">Analysis based on page content</div>
    `;
}