/**
 * Browser Extension Popup JavaScript
 * Author: UWU/CST/21/083
 */

document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const currentUrlElement = document.getElementById('currentUrl');
    const refreshBtn = document.getElementById('refreshBtn');
    const statusCard = document.getElementById('statusCard');
    const statusIcon = document.getElementById('statusIcon');
    const statusTitle = document.getElementById('statusTitle');
    const riskScoreElement = document.getElementById('riskScore');
    const riskLevelElement = document.getElementById('riskLevel');
    const statusMessage = document.getElementById('statusMessage');
    const domainValue = document.getElementById('domainValue');
    const lengthValue = document.getElementById('lengthValue');
    const issuesValue = document.getElementById('issuesValue');
    const timeValue = document.getElementById('timeValue');
    const issuesList = document.getElementById('issuesList');
    const issuesSection = document.getElementById('issuesSection');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const settingsBtn = document.getElementById('settingsBtn');
    const scannedTodayElement = document.getElementById('scannedToday');
    
    // State
    let currentTab = null;
    let analysisResult = null;
    
    // Initialize
    init();
    
    async function init() {
        // Get current tab
        const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
        currentTab = tabs[0];
        
        // Display current URL
        updateCurrentUrl(currentTab.url);
        
        // Load stats
        loadStats();
        
        // Analyze current URL
        analyzeCurrentUrl();
        
        // Set up event listeners
        setupEventListeners();
    }
    
    function updateCurrentUrl(url) {
        const urlText = url || 'No URL available';
        const urlSpan = currentUrlElement.querySelector('span');
        urlSpan.textContent = urlText;
        urlSpan.title = urlText;
    }
    
    async function loadStats() {
        const stats = await chrome.storage.local.get(['scannedToday']);
        scannedTodayElement.textContent = stats.scannedToday || '0';
    }
    
    async function analyzeCurrentUrl() {
        if (!currentTab || !currentTab.url) {
            showError('No URL to analyze');
            return;
        }
        
        // Show loading state
        showLoading();
        
        try {
            // Get analysis from background script
            const response = await chrome.runtime.sendMessage({
                action: 'analyzeUrl',
                url: currentTab.url
            });
            
            if (response.success) {
                analysisResult = response.result;
                updateUI(analysisResult);
                updateStats();
            } else {
                throw new Error(response.error || 'Analysis failed');
            }
        } catch (error) {
            console.error('Analysis error:', error);
            showError('Failed to analyze URL');
        }
    }
    
    function showLoading() {
        statusCard.className = 'status-card';
        statusIcon.className = 'fas fa-spinner fa-spin';
        statusTitle.textContent = 'Analyzing...';
        riskScoreElement.textContent = '...';
        riskLevelElement.textContent = 'LOADING';
        statusMessage.textContent = 'Checking URL for security threats...';
        
        // Clear details
        domainValue.textContent = '-';
        lengthValue.textContent = '-';
        issuesValue.textContent = '0';
        timeValue.textContent = '-';
        
        // Clear issues
        issuesList.innerHTML = `
            <div class="no-issues">
                <i class="fas fa-spinner fa-spin"></i>
                <p>Analyzing...</p>
            </div>
        `;
    }
    
    function showError(message) {
        statusCard.className = 'status-card warning';
        statusIcon.className = 'fas fa-exclamation-triangle';
        statusTitle.textContent = 'Error';
        riskScoreElement.textContent = '?';
        riskLevelElement.textContent = 'ERROR';
        statusMessage.textContent = message;
    }
    
    function updateUI(result) {
        const riskLevel = result.risk_level.toLowerCase();
        const riskScore = result.risk_score;
        const issues = result.issues || [];
        
        // Update status card
        statusCard.className = `status-card ${riskLevel}`;
        
        // Update icon based on risk level
        let iconClass, title, message;
        
        switch(riskLevel) {
            case 'safe':
            case 'low':
                iconClass = 'fas fa-check-circle';
                title = 'Safe URL';
                message = 'This URL appears to be safe.';
                break;
            case 'warning':
            case 'medium':
                iconClass = 'fas fa-exclamation-triangle';
                title = 'Warning';
                message = 'This URL has some suspicious characteristics.';
                break;
            case 'danger':
            case 'high':
                iconClass = 'fas fa-radiation';
                title = 'Danger';
                message = 'This URL is potentially malicious!';
                break;
            case 'critical':
                iconClass = 'fas fa-skull-crossbones';
                title = 'Critical Threat';
                message = 'DANGER: This URL shows clear malicious signs!';
                break;
            default:
                iconClass = 'fas fa-question-circle';
                title = 'Unknown';
                message = 'Unable to determine safety level.';
        }
        
        statusIcon.className = iconClass;
        statusTitle.textContent = title;
        riskScoreElement.textContent = riskScore.toFixed(1);
        riskLevelElement.textContent = riskLevel.toUpperCase();
        statusMessage.textContent = `${message} Risk score: ${riskScore.toFixed(1)}`;
        
        // Update details
        domainValue.textContent = result.domain || 'Unknown';
        lengthValue.textContent = result.url_length ? `${result.url_length} chars` : '-';
        issuesValue.textContent = issues.length;
        timeValue.textContent = result.analysis_time ? `${result.analysis_time}ms` : '-';
        
        // Update issues list
        updateIssuesList(issues);
    }
    
    function updateIssuesList(issues) {
        if (!issues || issues.length === 0) {
            issuesList.innerHTML = `
                <div class="no-issues">
                    <i class="fas fa-check"></i>
                    <p>No security issues detected</p>
                </div>
            `;
            issuesSection.style.display = 'none';
            return;
        }
        
        issuesSection.style.display = 'block';
        
        let issuesHTML = '';
        issues.forEach((issue, index) => {
            issuesHTML += `
                <div class="issue-item">
                    <i class="fas fa-exclamation-circle"></i>
                    <div class="issue-text">${issue}</div>
                </div>
            `;
        });
        
        issuesList.innerHTML = issuesHTML;
    }
    
    async function updateStats() {
        const today = new Date().toDateString();
        const stats = await chrome.storage.local.get(['scannedToday', 'lastScanDate']);
        
        if (stats.lastScanDate === today) {
            // Same day, increment count
            const newCount = (stats.scannedToday || 0) + 1;
            await chrome.storage.local.set({
                scannedToday: newCount,
                lastScanDate: today
            });
            scannedTodayElement.textContent = newCount;
        } else {
            // New day, reset count
            await chrome.storage.local.set({
                scannedToday: 1,
                lastScanDate: today
            });
            scannedTodayElement.textContent = '1';
        }
    }
    
    function setupEventListeners() {
        // Refresh button
        refreshBtn.addEventListener('click', analyzeCurrentUrl);
        
        // Analyze button
        analyzeBtn.addEventListener('click', analyzeCurrentUrl);
        
        // Settings button
        settingsBtn.addEventListener('click', function() {
            chrome.runtime.openOptionsPage();
        });
        
        // Listen for messages from content script
        chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
            if (request.action === 'urlChanged') {
                updateCurrentUrl(request.url);
                analyzeCurrentUrl();
            }
        });
    }
});