/**
 * AI Avatar - Voice Search
 * Frontend JavaScript
 */

// Global variables for audio recording
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

// DOM elements
const questionInput = document.getElementById('questionInput');
const submitTextBtn = document.getElementById('submitTextBtn');
const recordButton = document.getElementById('recordButton');
const resultsContainer = document.getElementById('results');
const loadingIndicator = document.getElementById('loading-indicator');
const errorMessage = document.getElementById('error-message');
const statusMessage = document.getElementById('status-message');
const webInput = document.getElementById('web-input');
const addKnowledgeBtn = document.getElementById('add-knowledge-btn');
const documentsList = document.getElementById('documents-list');

// Event listeners
submitTextBtn.addEventListener('click', () => submitTextQuery(questionInput.value));
recordButton.addEventListener('mousedown', startRecording);
recordButton.addEventListener('mouseup', stopRecording);
recordButton.addEventListener('mouseleave', stopRecording);

// Add Knowledge button click handler
addKnowledgeBtn.addEventListener('click', async () => {
    await processWebPage(webInput.value);
    await loadIndexedDocuments(); // Reload the documents list
});

// Load indexed documents on page load
document.addEventListener('DOMContentLoaded', loadIndexedDocuments);

// Handle keyboard shortcuts for textarea
questionInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault(); // Prevent default newline
        submitTextQuery(this.value);
    }
});

// Toggle between text and voice input
document.getElementById('textInputBtn').addEventListener('click', function() {
    this.classList.add('active');
    document.getElementById('voiceInputBtn').classList.remove('active');
    document.getElementById('textInputSection').style.display = 'block';
    document.getElementById('voiceInputSection').style.display = 'none';
});

document.getElementById('voiceInputBtn').addEventListener('click', function() {
    this.classList.add('active');
    document.getElementById('textInputBtn').classList.remove('active');
    document.getElementById('textInputSection').style.display = 'none';
    document.getElementById('voiceInputSection').style.display = 'block';
});

// Check system status periodically
setInterval(checkSystemStatus, 30000);
checkSystemStatus(); // Initial check

// Functions
function showLoading() {
    loadingIndicator.style.display = 'block';
    errorMessage.style.display = 'none';
    resultsContainer.style.display = 'none';
}

function hideLoading() {
    loadingIndicator.style.display = 'none';
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
    resultsContainer.style.display = 'none';
}

// Helper function to format source path
function formatSourcePath(url) {
    try {
        const urlObj = new URL(url);
        return urlObj.pathname.split('/').pop().replace(/-/g, ' ');
    } catch (e) {
        return url;
    }
}

function showResults(data) {
    const resultsContainer = document.getElementById('results');
    const loadingIndicator = document.getElementById('loading-indicator');
    const errorMessage = document.getElementById('error-message');
    
    // Hide loading indicator and error message
    loadingIndicator.style.display = 'none';
    errorMessage.style.display = 'none';
    
    if (data.error) {
        errorMessage.textContent = data.error;
        errorMessage.style.display = 'block';
        return;
    }
    
    // Create HTML for the answer with markdown rendering
    let html = `
        <div class="answer-section">
            <h3>Answer</h3>
            <div class="markdown-content">${marked.parse(data.answer)}</div>
        </div>
    `;
    
    // Add sources section if there are any
    if (data.sources && data.sources.length > 0) {
        html += `
            <div class="sources-section">
                <h3>Sources</h3>
                ${data.sources.map(source => `
                    <div class="source-item">
                        <div class="source-title">
                            <a href="${source.url}" target="_blank" rel="noopener noreferrer">
                                ${source.title || formatSourcePath(source.url)}
                            </a>
                        </div>
                        <div class="source-text">
                            ${source.text}
                        </div>
                        <div class="source-score">
                            Relevance: ${(source.score * 100).toFixed(1)}%
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    }
    
    // Update the results container
    resultsContainer.innerHTML = html;
    
    // Show the results
    resultsContainer.style.display = 'block';
}

// Voice recording functions
async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        
        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };
        
        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            await submitVoiceQuery(audioBlob);
        };
        
        mediaRecorder.start();
        isRecording = true;
        recordButton.classList.add('recording');
        recordButton.querySelector('.record-text').textContent = 'Release to Stop';
    } catch (error) {
        showError('Error accessing microphone: ' + error.message);
    }
}

function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        recordButton.classList.remove('recording');
        recordButton.querySelector('.record-text').textContent = 'Hold to Record';
    }
}

async function submitVoiceQuery(audioBlob) {
    showLoading();
    
    try {
        const formData = new FormData();
        formData.append('audio', audioBlob);
        
        const response = await fetch('/transcribe', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Failed to transcribe audio');
        }
        
        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);
        }
        
        questionInput.value = data.text;
        await submitTextQuery(data.text);
    } catch (error) {
        showError('Error processing voice input: ' + error.message);
    } finally {
        hideLoading();
    }
}

async function submitTextQuery(query) {
    if (!query.trim()) {
        showError('Please enter a question');
        return;
    }
    
    showLoading();
    
    try {
        const response = await fetch('/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query })
        });
        
        const data = await response.json();
        
        if (data.error) {
            // If we have sources but no answer, show them
            if (data.sources && data.sources.length > 0) {
                showResults(
                    "I couldn't find a direct answer to your question, but here are some relevant sections from the content:",
                    data.sources
                );
            } else {
                showError(data.error);
            }
            return;
        }
        
        if (!response.ok) {
            throw new Error('Failed to process query');
        }
        
        // Show the answer and sources
        const answer = data.answer || "No answer found.";
        const sources = data.sources || [];  // Use empty array if sources is undefined
        showResults(data);
    } catch (error) {
        showError('Error processing query: ' + error.message);
    } finally {
        hideLoading();
    }
}

// Function to load indexed documents
async function loadIndexedDocuments() {
    try {
        documentsList.innerHTML = '<div class="loading">Loading documents...</div>';
        
        const response = await fetch('/indexed_documents');
        if (!response.ok) {
            throw new Error('Failed to fetch documents');
        }
        
        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);
        }
        
        if (!data.documents || data.documents.length === 0) {
            documentsList.innerHTML = '<div class="no-documents">No documents indexed yet</div>';
            return;
        }
        
        let html = '';
        data.documents.forEach(doc => {
            html += `
                <div class="document-item">
                    <span class="document-title">${doc.title}</span>
                    <a href="/data/raw/${doc.url}" target="_blank" class="document-url">${doc.url}</a>
                    ${doc.last_updated ? `<span class="document-date">Last updated: ${doc.last_updated}</span>` : ''}
                </div>
            `;
        });
        
        documentsList.innerHTML = html;
    } catch (error) {
        console.error('Error loading indexed documents:', error);
        documentsList.innerHTML = `<div class="error-message">Error loading documents: ${error.message}</div>`;
    }
}

async function processWebPage(url) {
    if (!url.trim()) {
        showError('Please enter a URL');
        return;
    }
    
    showLoading();
    
    try {
        const response = await fetch('/process_web', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ url })
        });
        
        if (!response.ok) {
            throw new Error('Failed to process web page');
        }
        
        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Reload the indexed documents list
        await loadIndexedDocuments();
        
        // Show success message
        const resultsContainer = document.getElementById('results');
        const loadingIndicator = document.getElementById('loading-indicator');
        const errorMessage = document.getElementById('error-message');
        
        // Hide loading indicator and error message
        loadingIndicator.style.display = 'none';
        errorMessage.style.display = 'none';
        
        // Show success message
        resultsContainer.innerHTML = `
            <div class="status-message success">
                <span class="material-icons">check_circle</span>
                <p>Web page processed successfully</p>
            </div>
        `;
        resultsContainer.style.display = 'block';
        
        webInput.value = '';
    } catch (error) {
        showError('Error processing web page: ' + error.message);
    } finally {
        hideLoading();
    }
}

async function checkSystemStatus() {
    try {
        const response = await fetch('/status');
        if (!response.ok) {
            throw new Error('Failed to check system status');
        }
        
        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);
        }
        
        statusMessage.textContent = `System Status: ${data.status}`;
        statusMessage.className = `status-message ${data.status}`;
        
        // Update UI based on status
        recordButton.disabled = !data.ollama_available;
        submitTextBtn.disabled = !data.index_working;
        addKnowledgeBtn.disabled = !data.ollama_available;
        
        if (!data.ollama_available) {
            statusMessage.textContent += ' (Voice input disabled)';
        }
        if (!data.index_working) {
            statusMessage.textContent += ' (Search disabled)';
        }
    } catch (error) {
        statusMessage.textContent = 'System Status: Error';
        statusMessage.className = 'status-message error';
    }
} 