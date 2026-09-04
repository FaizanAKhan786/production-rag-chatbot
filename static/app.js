// RAG Chatbot - Frontend Application
// No framework, no dependencies

const API_URL = '/chat';
const MAX_QUERY_LENGTH = 1000;

// DOM elements
const chatArea = document.getElementById('chatArea');
const queryInput = document.getElementById('queryInput');
const sendBtn = document.getElementById('sendBtn');
const statusEl = document.getElementById('status');

let isStreaming = false;

// --- Message rendering ---

function createMessageElement(role, content = '') {
    const msg = document.createElement('div');
    msg.className = `message message-${role}`;
    
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    
    if (role === 'assistant') {
        const textEl = document.createElement('div');
        textEl.className = 'message-text';
        textEl.textContent = content;
        bubble.appendChild(textEl);
        
        // Placeholder for citations
        const citationsEl = document.createElement('div');
        citationsEl.className = 'citations-container';
        citationsEl.style.display = 'none';
        bubble.appendChild(citationsEl);
    } else {
        bubble.textContent = content;
    }
    
    msg.appendChild(bubble);
    return msg;
}

function addUserMessage(text) {
    // Remove welcome message if present
    const welcome = chatArea.querySelector('.welcome-message');
    if (welcome) welcome.remove();
    
    const msg = createMessageElement('user', text);
    chatArea.appendChild(msg);
    scrollToBottom();
}

function addAssistantMessage() {
    const msg = createMessageElement('assistant');
    chatArea.appendChild(msg);
    
    // Add loading dots
    const textEl = msg.querySelector('.message-text');
    textEl.innerHTML = '<span class="loading-dots"><span>.</span><span>.</span><span>.</span></span>';
    
    scrollToBottom();
    return msg;
}

function appendToken(msgEl, token) {
    const textEl = msgEl.querySelector('.message-text');
    
    // Remove loading dots on first token
    const dots = textEl.querySelector('.loading-dots');
    if (dots) {
        textEl.textContent = '';
    }
    
    textEl.textContent += token;
    scrollToBottom();
}

function renderCitations(msgEl, citations) {
    if (!citations || citations.length === 0) return;
    
    const container = msgEl.querySelector('.citations-container');
    container.style.display = 'block';
    
    const header = document.createElement('button');
    header.className = 'citations-toggle';
    header.textContent = `📎 ${citations.length} source${citations.length > 1 ? 's' : ''} cited`;
    header.onclick = () => {
        const list = container.querySelector('.citations-list');
        list.classList.toggle('expanded');
        header.classList.toggle('expanded');
    };
    container.appendChild(header);
    
    const list = document.createElement('div');
    list.className = 'citations-list';
    
    for (const cite of citations) {
        const card = document.createElement('div');
        card.className = 'citation-card';
        
        const title = document.createElement('div');
        title.className = 'citation-title';
        title.textContent = `[${cite.citation_number}] ${cite.source_file}`;
        if (cite.page && cite.page > 0) {
            title.textContent += ` · p.${cite.page}`;
        }
        card.appendChild(title);
        
        if (cite.section) {
            const section = document.createElement('div');
            section.className = 'citation-section';
            section.textContent = cite.section;
            card.appendChild(section);
        }
        
        const preview = document.createElement('div');
        preview.className = 'citation-preview';
        preview.textContent = cite.text_preview;
        card.appendChild(preview);
        
        const score = document.createElement('div');
        score.className = 'citation-score';
        score.textContent = `Relevance: ${(cite.score * 100).toFixed(1)}%`;
        card.appendChild(score);
        
        list.appendChild(card);
    }
    
    container.appendChild(list);
}

function showError(msgEl, errorMsg) {
    const textEl = msgEl.querySelector('.message-text');
    textEl.textContent = '';
    textEl.className = 'message-text error';
    textEl.textContent = errorMsg;
}

function scrollToBottom() {
    chatArea.scrollTop = chatArea.scrollHeight;
}

// --- SSE Streaming ---

async function sendQuery(query) {
    if (isStreaming) return;
    if (!query.trim()) return;
    if (query.length > MAX_QUERY_LENGTH) {
        alert(`Query too long. Maximum ${MAX_QUERY_LENGTH} characters.`);
        return;
    }
    
    isStreaming = true;
    setStatus('Thinking...');
    sendBtn.disabled = true;
    queryInput.disabled = true;
    
    addUserMessage(query);
    const assistantMsg = addAssistantMessage();
    
    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query }),
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Server error: ${response.status}`);
        }
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        
        setStatus('Streaming...');
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            
            // Parse SSE events from buffer
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';  // Keep incomplete line in buffer
            
            for (const line of lines) {
                if (line.startsWith('event: ')) {
                    // Store event type for next data line
                    assistantMsg._currentEvent = line.slice(7).trim();
                } else if (line.startsWith('data: ')) {
                    const data = line.slice(6);
                    const eventType = assistantMsg._currentEvent || 'token';
                    
                    if (eventType === 'token') {
                        appendToken(assistantMsg, data);
                    } else if (eventType === 'done') {
                        try {
                            const doneData = JSON.parse(data);
                            renderCitations(assistantMsg, doneData.citations);
                            
                            if (doneData.no_context) {
                                assistantMsg.querySelector('.bubble').classList.add('no-context');
                            }
                        } catch (e) {
                            console.error('Failed to parse done event:', e);
                        }
                    } else if (eventType === 'error') {
                        try {
                            const errData = JSON.parse(data);
                            showError(assistantMsg, errData.detail || errData.error);
                        } catch (e) {
                            showError(assistantMsg, data);
                        }
                    }
                    
                    assistantMsg._currentEvent = null;
                }
            }
        }
        
        setStatus('Ready');
        
    } catch (error) {
        console.error('Chat error:', error);
        showError(assistantMsg, error.message || 'Failed to connect to server.');
        setStatus('Error');
    } finally {
        isStreaming = false;
        sendBtn.disabled = false;
        queryInput.disabled = false;
        queryInput.focus();
    }
}

// --- UI Helpers ---

function setStatus(text) {
    statusEl.textContent = text;
    statusEl.className = 'status' + (text === 'Error' ? ' error' : '');
}

// Auto-resize textarea
function autoResize() {
    queryInput.style.height = 'auto';
    queryInput.style.height = Math.min(queryInput.scrollHeight, 120) + 'px';
}

// --- Event Listeners ---

sendBtn.addEventListener('click', () => {
    sendQuery(queryInput.value);
    queryInput.value = '';
    autoResize();
});

queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendQuery(queryInput.value);
        queryInput.value = '';
        autoResize();
    }
});

queryInput.addEventListener('input', autoResize);

// Focus input on load
queryInput.focus();
