/* NEXUS AI | v6.0 Premium Logic */

let tickerList = [];
let selectedTicker = "";
let activeChatTicker = null;
let priceChartInstance = null;
let trajChartInstance = null;

document.addEventListener('DOMContentLoaded', () => {
    loadTickers();
    setupSearchEvents();
    setupChatEvents();
    setupChatResizer();
});

// --- 1. SEARCH LOGIC ---
async function loadTickers() {
    try {
        const response = await fetch('/static/tickers.json');
        if (!response.ok) throw new Error("File not found");
        tickerList = await response.json();
        console.log(`✅ Loaded ${tickerList.length} assets`);
    } catch (error) { console.warn("⚠️ Tickers missing."); }
}

function setupSearchEvents() {
    const input = document.getElementById('search-input');
    const box = document.getElementById('suggestions');
    const btn = document.getElementById('analyze-btn');

    input.addEventListener('input', function() {
        const query = this.value.toUpperCase().trim();
        if (query.length < 1) { box.classList.add('hidden'); return; }
        
        let matches = tickerList.filter(item => item.ticker.startsWith(query) || item.title.toUpperCase().includes(query));
        
        // Sorting Logic: Exact Ticker > Starts With Ticker > Title Match
        matches.sort((a, b) => {
            const aTicker = a.ticker.toUpperCase();
            const bTicker = b.ticker.toUpperCase();
            
            // 1. Exact Match Priority
            if (aTicker === query) return -1;
            if (bTicker === query) return 1;

            // 2. Starts With Priority
            const aStarts = aTicker.startsWith(query);
            const bStarts = bTicker.startsWith(query);
            if (aStarts && !bStarts) return -1;
            if (!aStarts && bStarts) return 1;

            return 0;
        });

        renderSuggestions(matches.slice(0, 5));
    });

    document.addEventListener('click', (e) => {
        if (!e.target.closest('.search-wrapper')) box.classList.add('hidden');
    });

    btn.addEventListener('click', () => runAnalysis(input.value));
    
    input.addEventListener('keypress', (e) => {
        if(e.key === 'Enter') {
            const val = input.value.toUpperCase().trim();
            // Same sorting logic for Enter key selection to ensure consistency
            let matches = tickerList.filter(item => item.ticker.startsWith(val) || item.title.toUpperCase().includes(val));
            
            matches.sort((a, b) => {
                if (a.ticker === val) return -1;
                if (b.ticker === val) return 1;
                if (a.ticker.startsWith(val) && !b.ticker.startsWith(val)) return -1;
                if (!a.ticker.startsWith(val) && b.ticker.startsWith(val)) return 1;
                return 0;
            });

            if(matches.length > 0) {
                const best = matches[0];
                input.value = best.ticker;
                box.classList.add('hidden');
                runAnalysis(best.ticker);
            } else {
                runAnalysis(input.value);
            }
        }
    });
}

function renderSuggestions(matches) {
    const box = document.getElementById('suggestions');
    box.innerHTML = '';
    if (matches.length === 0) { box.classList.add('hidden'); return; }
    
    matches.forEach(item => {
        const div = document.createElement('div');
        div.className = 'suggestion-item';
        div.innerHTML = `<span class="s-ticker">${item.ticker}</span><span class="s-name">${item.title}</span>`;
        div.onclick = () => {
            document.getElementById('search-input').value = item.ticker;
            box.classList.add('hidden');
            runAnalysis(item.ticker);
        };
        box.appendChild(div);
    });
    box.classList.remove('hidden');
}

// --- 2. ANALYSIS ---
async function runAnalysis(inputVal) {
    if(!inputVal) return;
    let ticker = inputVal.split(' - ')[0].trim().toUpperCase();

    document.getElementById('results').classList.add('hidden');
    document.getElementById('loader').classList.remove('hidden');

    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ ticker: ticker, company: "", ceo: "" })
        });
        const data = await response.json();
        if(data.error) throw new Error(data.error);

        displayResults(data);
        if (data.price_data && data.price_data.chart_data) renderCharts(data.price_data.chart_data);
        enableChat(ticker);

    } catch (error) {
        alert("Analysis Error: " + error.message);
    } finally {
        document.getElementById('loader').classList.add('hidden');
    }
}

function displayResults(data) {
    document.getElementById('results').classList.remove('hidden');
    
    // Signal
    const sigEl = document.getElementById('final-signal');
    sigEl.innerHTML = `${data.emoji} ${data.signal}`;
    sigEl.className = 'big-signal ' + (data.score > 0 ? 'text-green' : data.score < 0 ? 'text-red' : '');
    // Confidence removed

    // Price
    if(data.price_data) {
        const p = data.price_data;
        document.getElementById('price-current').innerText = `$${p.current_price.toFixed(2)}`;
        document.getElementById('price-pred').innerText = `$${p.predicted_price.toFixed(2)}`;
        const chgEl = document.getElementById('price-change');
        chgEl.innerText = `${p.pct_change > 0 ? '+' : ''}${p.pct_change.toFixed(2)}%`;
        chgEl.className = 'metric-val ' + (p.pct_change > 0 ? 'text-green' : 'text-red');
    }

    // Sentiment Bars
    const s = data.news_data.stats;
    const total = s.positive + s.neutral + s.negative || 1;
    document.getElementById('bar-pos').style.width = `${(s.positive/total)*100}%`;
    document.getElementById('val-pos').innerText = `${Math.round((s.positive/total)*100)}%`;
    document.getElementById('bar-neu').style.width = `${(s.neutral/total)*100}%`;
    document.getElementById('val-neu').innerText = `${Math.round((s.neutral/total)*100)}%`;
    document.getElementById('bar-neg').style.width = `${(s.negative/total)*100}%`;
    document.getElementById('val-neg').innerText = `${Math.round((s.negative/total)*100)}%`;
}

// --- 3. CHARTS ---
function renderCharts(chartData) {
    const commonOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { labels: { color: '#888', font: { family: 'Inter', size: 11 } } },
            tooltip: {
                backgroundColor: 'rgba(15, 17, 21, 0.9)',
                titleColor: '#D4AF37',
                bodyColor: '#fff',
                borderColor: 'rgba(255,255,255,0.1)',
                borderWidth: 1,
                padding: 10,
                displayColors: false
            }
        },
        scales: {
            x: { grid: { display: false }, ticks: { display: false } },
            y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#555', font: { family: 'JetBrains Mono', size: 10 } } }
        },
        interaction: { intersect: false, mode: 'index' }
    };

    // PRICE CHART
    const ctxPrice = document.getElementById('priceChart').getContext('2d');
    if(priceChartInstance) priceChartInstance.destroy();

    const gradientHist = ctxPrice.createLinearGradient(0, 0, 0, 400);
    gradientHist.addColorStop(0, 'rgba(34, 197, 94, 0.2)');
    gradientHist.addColorStop(1, 'rgba(34, 197, 94, 0)');

    const histLen = chartData.history_prices.length;
    const labels = [...chartData.history_dates, ...chartData.proj_dates];
    const dataHist = [...chartData.history_prices, null, null, null];
    const dataProj = Array(histLen - 1).fill(null);
    dataProj.push(chartData.history_prices[histLen - 1], ...chartData.proj_prices);

    priceChartInstance = new Chart(ctxPrice, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Historical',
                    data: dataHist,
                    borderColor: '#22c55e',
                    backgroundColor: gradientHist,
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: true,
                    tension: 0.3
                },
                {
                    label: 'AI Projection',
                    data: dataProj,
                    borderColor: '#D4AF37',
                    borderWidth: 2,
                    borderDash: [4, 4],
                    pointRadius: 0,
                    tension: 0.3
                }
            ]
        },
        options: commonOptions
    });

    // MOMENTUM CHART
    const ctxTraj = document.getElementById('trajChart').getContext('2d');
    if(trajChartInstance) trajChartInstance.destroy();

    const histReturns = [];
    for(let i=1; i<chartData.history_prices.length; i++) {
        histReturns.push(((chartData.history_prices[i] - chartData.history_prices[i-1])/chartData.history_prices[i-1]) * 100);
    }
    let prev = chartData.history_prices[histLen-1];
    const projReturns = [];
    chartData.proj_prices.forEach(p => {
        projReturns.push(((p - prev)/prev) * 100);
        prev = p;
    });

    const trajData = [...histReturns, ...projReturns];
    const barColors = trajData.map((v, i) => i >= histReturns.length ? '#D4AF37' : (v >= 0 ? '#22c55e' : '#ef4444'));

    trajChartInstance = new Chart(ctxTraj, {
        type: 'bar',
        data: {
            labels: labels.slice(1),
            datasets: [{
                label: 'Momentum %',
                data: trajData,
                backgroundColor: barColors,
                borderRadius: 2,
                barThickness: 'flex',
                maxBarThickness: 8
            }]
        },
        options: {
            ...commonOptions,
            plugins: { legend: { display: false } },
            scales: { x: { display: false }, y: { display: false } }
        }
    });
}

// --- 4. CHAT & RESIZE ---
function setupChatEvents() {
    const input = document.getElementById('chat-input');
    const btn = document.getElementById('chat-send');
    btn.onclick = () => sendQuery();
    input.onkeypress = (e) => { if(e.key === 'Enter') sendQuery(); };
}

function setupChatResizer() {
    const resizer = document.getElementById('chat-resizer');
    let isResizing = false;

    resizer.addEventListener('mousedown', (e) => {
        isResizing = true;
        document.body.style.cursor = 'ew-resize';
        e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
        if (!isResizing) return;
        let newWidth = window.innerWidth - e.clientX;
        if (newWidth < 300) newWidth = 300;
        if (newWidth > window.innerWidth * 0.8) newWidth = window.innerWidth * 0.8;
        
        document.documentElement.style.setProperty('--chat-width', `${newWidth}px`);
    });

    document.addEventListener('mouseup', () => {
        isResizing = false;
        document.body.style.cursor = 'default';
    });
}

function toggleChat() {
    document.getElementById('chat-panel').classList.toggle('open');
    document.getElementById('main-blur').classList.toggle('blur');
    document.getElementById('main-blur').classList.toggle('slide-left');
}

function enableChat(ticker) {
    activeChatTicker = ticker;
    document.getElementById('chat-input').disabled = false;
    document.getElementById('chat-send').disabled = false;
    document.getElementById('chat-messages').innerHTML = `<div class="msg msg-bot">Neural Link Established: ${ticker}. Data Loaded.</div>`;
}

async function sendQuery() {
    const input = document.getElementById('chat-input');
    const val = input.value.trim();
    if(!val || !activeChatTicker) return;

    addMsg(val, 'user');
    input.value = '';
    
    try {
        const res = await fetch('/api/chat', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ ticker: activeChatTicker, query: val })
        });
        const data = await res.json();
        addMsg(data.response || data.error, 'bot');
    } catch(e) { addMsg("Connection Failed", 'bot'); }
}

function addMsg(text, role) {
    const div = document.createElement('div');
    div.className = `msg msg-${role}`;
    div.innerHTML = text.replace(/\n/g, '<br>').replace(/\*\*(.*?)\*\*/g, '<strong style="color:var(--gold-primary)">$1</strong>');
    const box = document.getElementById('chat-messages');
    box.appendChild(div);
    box.scrollTop = box.scrollHeight;
}