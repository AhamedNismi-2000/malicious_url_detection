// Content Script - Simple version
console.log('✅ URL Safety Detector content script loaded');

// Listen for URL changes
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'urlChanged') {
        console.log('🌐 Page changed to:', request.url);
    }
    return true;
});