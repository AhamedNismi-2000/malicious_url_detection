// ============================================
// URL Safety Detector - Background Service Worker
// FIXED VERSION - NO NOTIFICATION ERRORS
// Author: UWU/CST/21/083
// ============================================

console.log('🔧 URL Safety Detector Background Script STARTING...');

// ============ 1. INITIALIZATION ============
chrome.runtime.onInstalled.addListener(() => {
    console.log('✅ Extension installed successfully');
    
    // Initialize default settings
    const defaultSettings = {
        scannedToday: 0,
        lastScanDate: new Date().toDateString(),
        settings: {
            heuristicEnabled: true,
            autoScan: false, // Disabled by default
            minRiskLevel: 'high'
        }
    };
    
    chrome.storage.local.set(defaultSettings, () => {
        console.log('✅ Default settings saved');
    });
});

// ============ 2. TAB MONITORING ============
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    if (changeInfo.status === 'complete' && tab.url) {
        console.log('📊 Tab loaded:', tab.url.substring(0, 60));
        
        // Auto-scan if enabled
        chrome.storage.local.get(['settings'], (data) => {
            if (data.settings?.autoScan) {
                analyzeUrl(tab.url).then(result => {
                    console.log(`Auto-scan: ${result.risk_level.toUpperCase()}`);
                });
            }
        });
        
        // Notify content script
        chrome.tabs.sendMessage(tabId, {
            action: 'urlChanged',
            url: tab.url
        }).catch(err => {
            // Content script not ready - normal on first load
        });
    }
});

// ============ 3. MESSAGE HANDLER ============
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    console.log('📩 Message received:', request.action);
    
    // 3.1 PING - Check if service worker is alive
    if (request.action === 'ping') {
        sendResponse({
            status: 'alive',
            time: new Date().toISOString(),
            version: '1.0.0'
        });
        return true;
    }
    
    // 3.2 ANALYZE URL - Main functionality
    if (request.action === 'analyzeUrl') {
        analyzeUrl(request.url)
            .then(result => {
                sendResponse({ success: true, result: result });
            })
            .catch(error => {
                sendResponse({ 
                    success: false, 
                    error: error.message 
                });
            });
        return true; // Required for async
    }
    
    // 3.3 GET SETTINGS
    if (request.action === 'getSettings') {
        chrome.storage.local.get(['settings'], (data) => {
            sendResponse(data.settings || {});
        });
        return true;
    }
    
    // 3.4 SAVE SETTINGS
    if (request.action === 'saveSettings') {
        chrome.storage.local.get(['settings'], (data) => {
            const newSettings = { ...data.settings, ...request.settings };
            chrome.storage.local.set({ settings: newSettings }, () => {
                sendResponse({ success: true });
            });
        });
        return true;
    }
    
    // Default response
    sendResponse({ error: 'Unknown action' });
    return true;
});

// ============ 4. URL ANALYSIS FUNCTION ============
async function analyzeUrl(url) {
    const startTime = Date.now();
    
    try {
        // Extract domain
        let domain = 'unknown';
        try {
            const urlObj = new URL(url);
            domain = urlObj.hostname;
        } catch {
            domain = 'invalid-url';
        }
        
        // Perform heuristic analysis
        const analysis = heuristicAnalysis(url);
        
        return {
            url: url,
            domain: domain,
            url_length: url.length,
            risk_score: analysis.score,
            risk_level: analysis.riskLevel,
            is_malicious: analysis.isMalicious,
            issues: analysis.issues,
            analysis_time: Date.now() - startTime
        };
        
    } catch (error) {
        console.error('Analysis error:', error);
        return {
            url: url,
            domain: 'error',
            risk_score: 0,
            risk_level: 'unknown',
            is_malicious: false,
            issues: ['Analysis error'],
            analysis_time: Date.now() - startTime
        };
    }
}

// ============ 5. HEURISTIC ANALYSIS LOGIC ============
function heuristicAnalysis(url) {
    let score = 0;
    const issues = [];
    
    // Rule 1: URL length
    if (url.length > 100) {
        score += 1.5;
        issues.push(`Long URL (${url.length} chars)`);
    }
    
    // Rule 2: IP address
    const ipPattern = /\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b/;
    if (ipPattern.test(url)) {
        score += 2.0;
        issues.push('Contains IP address');
    }
    
    // Rule 3: Suspicious keywords
    const keywords = ['login', 'secure', 'verify', 'bank', 'password', 'account'];
    const urlLower = url.toLowerCase();
    keywords.forEach(keyword => {
        if (urlLower.includes(keyword)) {
            score += 1.2;
            issues.push(`Contains "${keyword}"`);
        }
    });
    
    // Rule 4: Many dots
    const dotCount = (url.match(/\./g) || []).length;
    if (dotCount > 5) {
        score += 1.0;
        issues.push(`Many dots (${dotCount})`);
    }
    
    // Rule 5: Suspicious TLD
    const tlds = ['.xyz', '.top', '.loan', '.tk', '.ml', '.ga', '.cf'];
    tlds.forEach(tld => {
        if (url.includes(tld)) {
            score += 1.7;
            issues.push(`Suspicious TLD: ${tld}`);
        }
    });
    
    // Rule 6: Special characters in domain
    if (/[@#\$%&]/.test(url)) {
        score += 1.5;
        issues.push('Special characters in URL');
    }
    
    // Determine risk level
    let riskLevel, isMalicious;
    if (score >= 6) {
        riskLevel = 'critical';
        isMalicious = true;
    } else if (score >= 4) {
        riskLevel = 'danger';
        isMalicious = true;
    } else if (score >= 2) {
        riskLevel = 'warning';
        isMalicious = false;
    } else {
        riskLevel = 'safe';
        isMalicious = false;
    }
    
    return {
        score: Math.min(score, 10),
        riskLevel: riskLevel,
        isMalicious: isMalicious,
        issues: issues
    };
}

// ============ 6. KEEP ALIVE MECHANISM ============
// Method 1: Alarms (recommended)
chrome.alarms.create('heartbeat', { periodInMinutes: 1 });

chrome.alarms.onAlarm.addListener((alarm) => {
    if (alarm.name === 'heartbeat') {
        console.log('❤️ Heartbeat:', new Date().toLocaleTimeString());
    }
});

// Method 2: Periodic logging
setInterval(() => {
    console.log('🔄 Service worker active');
}, 30000);

// ============ 7. STARTUP ============
chrome.runtime.onStartup.addListener(() => {
    console.log('🚀 Extension starting up...');
});

console.log('✅ Background Script READY and RUNNING');