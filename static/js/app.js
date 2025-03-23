/**
 * AI Avatar - Voice Search
 * Frontend JavaScript
 */

// Global variables for audio recording
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

// DOM elements
const microphoneButton = document.getElementById('microphone-button');
const searchButton = document.getElementById('search-button');
const queryInput = document.getElementById('query-input');
const resultsContainer = document.getElementById('results-container');
const loadingIndicator = document.getElementById('loading-indicator');
const errorMessage = document.getElementById('error-message');
const statusMessage = document.getElementById('status-message');
const confluenceInput = document.getElementById('confluence-input');
const processConfluenceButton = document.getElementById('process-confluence-button');

// Event listeners
microphoneButton.addEventListener('click', toggleRecording);
searchButton.addEventListener('click', () => submitTextQuery(queryInput.value));
processConfluenceButton.addEventListener('click', () => processConfluencePage(confluenceInput.value));

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

function showResults(answer, sources) {
    resultsContainer.innerHTML = `
        <h3>Answer:</h3>
        <p>${answer}</p>
        <h3>Sources:</h3>
        <ul>
            ${sources.map(source => `
                <li>
                    <a href="${source.url}" target="_blank">${source.url}</a>
                    ${source.score ? `(Score: ${(source.score * 100).toFixed(1)}%)` : ''}
                </li>
            `).join('')}
        </ul>
    `;
    resultsContainer.style.display = 'block';
    errorMessage.style.display = 'none';
}

async function toggleRecording() {
    if (!isRecording) {
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
            microphoneButton.classList.add('recording');
            microphoneButton.querySelector('span').textContent = 'Stop Recording';
        } catch (error) {
            showError('Error accessing microphone: ' + error.message);
        }
    } else {
        mediaRecorder.stop();
        isRecording = false;
        microphoneButton.classList.remove('recording');
        microphoneButton.querySelector('span').textContent = 'Start Recording';
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
        
        queryInput.value = data.text;
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
        showResults(answer, sources);
    } catch (error) {
        showError('Error processing query: ' + error.message);
    } finally {
        hideLoading();
    }
}

async function processConfluencePage(url) {
    if (!url.trim()) {
        showError('Please enter a Confluence page URL');
        return;
    }
    
    showLoading();
    
    try {
        const response = await fetch('/process_confluence', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ url })
        });
        
        if (!response.ok) {
            throw new Error('Failed to process Confluence page');
        }
        
        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);
        }
        
        showResults('Confluence page processed successfully', []);
        confluenceInput.value = '';
    } catch (error) {
        showError('Error processing Confluence page: ' + error.message);
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
        microphoneButton.disabled = !data.ollama_available;
        searchButton.disabled = !data.index_working;
        processConfluenceButton.disabled = !data.ollama_available;
        
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