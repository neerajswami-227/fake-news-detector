// background.js - improved text extraction
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "analyzePage") {
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            const tab = tabs[0];
            chrome.scripting.executeScript({
                target: { tabId: tab.id },
                func: extractPageContent
            }, (results) => {
                if (results && results[0] && results[0].result) {
                    const pageText = results[0].result;
                    fetch('http://127.0.0.1:5000/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text: pageText })
                    })
                    .then(res => res.json())
                    .then(data => sendResponse(data))
                    .catch(err => sendResponse({ error: err.message }));
                } else {
                    sendResponse({ error: "Could not extract page content." });
                }
            });
        });
        return true;
    }
});

// Improved extraction function
function extractPageContent() {
    // Try to find article container first
    let articleElement = document.querySelector('article') ||
                         document.querySelector('[role="main"]') ||
                         document.querySelector('.main-content') ||
                         document.querySelector('.story-content') ||
                         document.querySelector('.article-body');
    
    let text = '';
    if (articleElement) {
        // Extract from article container
        const paragraphs = articleElement.querySelectorAll('p');
        text = Array.from(paragraphs).map(p => p.innerText).join(' ');
    } else {
        // Fallback: clone body, remove ads and noisy elements
        const clone = document.body.cloneNode(true);
        const unwanted = clone.querySelectorAll('script, style, nav, footer, header, aside, .ad, .advertisement, .ad-banner, .social-share, .related-stories, .sidebar, .comments, .cookie-notice, iframe');
        unwanted.forEach(el => el.remove());
        const paragraphs = clone.querySelectorAll('p');
        text = Array.from(paragraphs).map(p => p.innerText).join(' ');
    }
    
    // Clean up: remove extra spaces and limit length
    text = text.replace(/\s+/g, ' ').trim();
    text = text.substring(0, 3000);
    return text || "No readable content found.";
}