// 全局状态
const state = {
    version: 'v5.0',
    frames: [],
    roi: { x: 0, y: 0, width: 0, height: 0 },
    isSelecting: false,
    selectionStart: null,
    currentImageA: null,
    currentImageB: null
};

// DOM 元素
const elements = {
    frameASelect: document.getElementById('frameA'),
    frameBSelect: document.getElementById('frameB'),
    muSubInput: document.getElementById('muSub'),
    deltaYInput: document.getElementById('deltaY'),
    searchRadiusInput: document.getElementById('searchRadius'),
    compareBtn: document.getElementById('compareBtn'),
    canvasA: document.getElementById('canvasA'),
    canvasB: document.getElementById('canvasB'),
    versionSelect: document.getElementById('algoVersion'),
    roiX: document.getElementById('roiX'),
    roiY: document.getElementById('roiY'),
    roiW: document.getElementById('roiW'),
    roiH: document.getElementById('roiH'),
    resultScore: document.getElementById('resultScore'),
    resultConfidence: document.getElementById('resultConfidence'),
    resultDecision: document.getElementById('resultDecision'),
    resultDelta: document.getElementById('resultDelta'),
    resultRoi: document.getElementById('resultRoi'),
    resultMetrics: document.getElementById('resultMetrics'),
    resultRaw: document.getElementById('resultRaw'),
    versionTag: document.getElementById('resultVersion'),
    status: document.getElementById('status')
};

const metricLabels = {
    overlap_iou: 'Mask IoU',
    overlap_dice: 'Mask Dice',
    overlap_tiou: 'Mask tIoU',
    structure_ssim: 'Structure SSIM',
    layout_projection: 'Layout Projection',
    alignment_peak: 'POC Peak',
    alignment_psr: 'POC PSR',
    similarity: 'Similarity',
    core_similarity: 'Core Similarity',
    match_fraction: 'Match Fraction',
    stroke_width_penalty: 'Stroke Width Penalty',
    template_similarity: 'Template Similarity',
    orb_similarity: 'ORB Similarity'
};

// 工具函数
function setStatus(message, isError = false) {
    elements.status.textContent = message;
    elements.status.style.background = isError ? '#d32f2f' : '#007acc';
}

function updateROIDisplay() {
    elements.roiX.textContent = state.roi.x;
    elements.roiY.textContent = state.roi.y;
    elements.roiW.textContent = state.roi.width;
    elements.roiH.textContent = state.roi.height;
}

// API 调用
async function loadFrames() {
    try {
        setStatus('Loading frames...');
        const response = await fetch('/frames');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const data = await response.json();
        state.frames = data.frames;
        
        // 填充下拉列表
        elements.frameASelect.innerHTML = '<option value="">选择帧...</option>';
        elements.frameBSelect.innerHTML = '<option value="">选择帧...</option>';
        
        data.frames.forEach(frame => {
            const optionA = document.createElement('option');
            optionA.value = frame.name;
            optionA.textContent = frame.name;
            elements.frameASelect.appendChild(optionA);

            const optionB = document.createElement('option');
            optionB.value = frame.name;
            optionB.textContent = frame.name;
            elements.frameBSelect.appendChild(optionB);
        });
        
        setStatus(`Loaded ${data.count} frames`);
    } catch (error) {
        setStatus(`Error loading frames: ${error.message}`, true);
        console.error('Load frames error:', error);
    }
}

async function loadFrameImage(frameName, canvas) {
    try {
        const response = await fetch(`/frames/${frameName}/image`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const blob = await response.blob();
        const img = new Image();
        
        return new Promise((resolve, reject) => {
            img.onload = () => {
                const wrapper = canvas.parentElement;
                const wrapperWidth = wrapper.clientWidth;
                const wrapperHeight = wrapper.clientHeight;
                
                // 计算缩放比例以适应容器
                const scaleX = wrapperWidth / img.width;
                const scaleY = wrapperHeight / img.height;
                const scale = Math.min(scaleX, scaleY, 1); // 不放大,只缩小
                
                // 设置canvas的实际尺寸为原始图片尺寸
                canvas.width = img.width;
                canvas.height = img.height;
                
                // 设置canvas的显示尺寸
                canvas.style.width = (img.width * scale) + 'px';
                canvas.style.height = (img.height * scale) + 'px';
                
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0);
                
                // 存储缩放比例以便选区计算
                canvas.dataset.scale = scale;
                canvas.dataset.displayWidth = img.width * scale;
                canvas.dataset.displayHeight = img.height * scale;
                
                resolve(img);
            };
            img.onerror = reject;
            img.src = URL.createObjectURL(blob);
        });
    } catch (error) {
        throw new Error(`Failed to load image: ${error.message}`);
    }
}

async function compareFrames() {
    const frameA = elements.frameASelect.value;
    const frameB = elements.frameBSelect.value;
    const version = elements.versionSelect.value || state.version;
    state.version = version;
    
    if (!frameA || !frameB) {
        setStatus('Please select both frames', true);
        return;
    }
    
    if (state.roi.width === 0 || state.roi.height === 0) {
        setStatus('Please select a ROI on Frame A', true);
        return;
    }
    
    const requestData = {
        version,
        frame_a: frameA,
        frame_b: frameB,
        roi: {
            x: state.roi.x,
            y: state.roi.y,
            width: state.roi.width,
            height: state.roi.height
        },
        mu_sub: parseFloat(elements.muSubInput.value),
        delta_y: parseFloat(elements.deltaYInput.value),
        search_radius: parseInt(elements.searchRadiusInput.value)
    };
    
    try {
        setStatus('Computing similarity...');
        const response = await fetch('/compare', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP ${response.status}`);
        }
        
        const result = await response.json();
        displayResult(result);
        setStatus('Comparison complete');
    } catch (error) {
        setStatus(`Error: ${error.message}`, true);
        console.error('Compare error:', error);
    }
}

function displayResult(result) {
    const confidence = result.confidence ?? result.score ?? 0;
    elements.resultScore.textContent = (result.score ?? 0).toFixed(4);
    elements.resultConfidence.textContent = confidence.toFixed(4);
    elements.resultDecision.textContent = result.decision ?? '-';
    if (elements.versionTag) {
        elements.versionTag.textContent = result.version ?? state.version ?? '-';
    }
    elements.resultDelta.textContent = `(${result.dx ?? 0}, ${result.dy ?? 0})`;

    if (result.roi) {
        elements.resultRoi.textContent = `x=${result.roi.x}, y=${result.roi.y}, w=${result.roi.width}, h=${result.roi.height}`;
    } else {
        elements.resultRoi.textContent = 'n/a';
    }

    const metrics = result.metrics || {};
    const lines = Object.entries(metricLabels).map(([key, label]) => {
        if (metrics[key] === undefined || metrics[key] === null) {
            return `${label}: n/a`;
        }
        const pct = (metrics[key] * 100).toFixed(1);
        return `${label}: ${pct}%`;
    });
    elements.resultMetrics.textContent = lines.join('\n');

    elements.resultRaw.textContent = JSON.stringify(result, null, 2);
}

// 在Frame B上绘制选区
function drawSelectionOnFrameB() {
    if (!state.currentImageB) return;
    
    const canvasB = elements.canvasB;
    const ctxB = canvasB.getContext('2d');
    
    // 重绘图像
    ctxB.clearRect(0, 0, canvasB.width, canvasB.height);
    ctxB.drawImage(state.currentImageB, 0, 0);
    
    // 绘制选区(使用与Frame A相同的坐标)
    if (state.roi.width > 0 && state.roi.height > 0) {
        ctxB.strokeStyle = '#0e639c';
        ctxB.lineWidth = 2;
        ctxB.strokeRect(state.roi.x, state.roi.y, state.roi.width, state.roi.height);
        
        ctxB.fillStyle = 'rgba(14, 99, 156, 0.2)';
        ctxB.fillRect(state.roi.x, state.roi.y, state.roi.width, state.roi.height);
    }
}

// Canvas 选区功能
function setupCanvasSelection() {
    const canvas = elements.canvasA;
    const ctx = canvas.getContext('2d');
    
    function getMousePos(e) {
        const rect = canvas.getBoundingClientRect();
        const scale = parseFloat(canvas.dataset.scale) || 1;
        
        // 计算鼠标在canvas上的位置(相对于显示尺寸)
        const x = (e.clientX - rect.left);
        const y = (e.clientY - rect.top);
        
        // 转换为原始图片坐标
        return {
            x: Math.floor(x / scale),
            y: Math.floor(y / scale)
        };
    }
    
    function drawSelection() {
        // 绘制Frame A的选区
        if (state.currentImageA) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(state.currentImageA, 0, 0);
            
            if (state.roi.width > 0 && state.roi.height > 0) {
                ctx.strokeStyle = '#0e639c';
                ctx.lineWidth = 2;
                ctx.strokeRect(state.roi.x, state.roi.y, state.roi.width, state.roi.height);
                
                ctx.fillStyle = 'rgba(14, 99, 156, 0.2)';
                ctx.fillRect(state.roi.x, state.roi.y, state.roi.width, state.roi.height);
            }
        }
        
        // 同步绘制Frame B的选区
        drawSelectionOnFrameB();
    }
    
    canvas.addEventListener('mousedown', (e) => {
        const pos = getMousePos(e);
        // 确保点击在图片范围内
        if (pos.x < 0 || pos.y < 0 || pos.x >= canvas.width || pos.y >= canvas.height) {
            return;
        }
        state.isSelecting = true;
        state.selectionStart = pos;
        state.roi = { x: pos.x, y: pos.y, width: 0, height: 0 };
        updateROIDisplay();
    });
    
    canvas.addEventListener('mousemove', (e) => {
        if (!state.isSelecting) return;
        
        const pos = getMousePos(e);
        // 限制在图片范围内
        const clampedX = Math.max(0, Math.min(pos.x, canvas.width - 1));
        const clampedY = Math.max(0, Math.min(pos.y, canvas.height - 1));
        
        const x = Math.min(state.selectionStart.x, clampedX);
        const y = Math.min(state.selectionStart.y, clampedY);
        const width = Math.abs(clampedX - state.selectionStart.x);
        const height = Math.abs(clampedY - state.selectionStart.y);
        
        state.roi = { x, y, width, height };
        updateROIDisplay();
        drawSelection();
    });
    
    canvas.addEventListener('mouseup', () => {
        state.isSelecting = false;
        drawSelection();
    });
    
    canvas.addEventListener('mouseleave', () => {
        if (state.isSelecting) {
            state.isSelecting = false;
            drawSelection();
        }
    });
}

// 事件监听器
elements.compareBtn.addEventListener('click', compareFrames);

elements.frameASelect.addEventListener('change', async (e) => {
    if (!e.target.value) return;
    try {
        setStatus('Loading Frame A...');
        state.currentImageA = await loadFrameImage(e.target.value, elements.canvasA);
        state.roi = { x: 0, y: 0, width: 0, height: 0 };
        updateROIDisplay();
        setStatus('Frame A loaded');
    } catch (error) {
        setStatus(`Error loading Frame A: ${error.message}`, true);
    }
});

elements.frameBSelect.addEventListener('change', async (e) => {
    if (!e.target.value) return;
    try {
        setStatus('Loading Frame B...');
        state.currentImageB = await loadFrameImage(e.target.value, elements.canvasB);
        // 加载完成后绘制选区
        drawSelectionOnFrameB();
        setStatus('Frame B loaded');
    } catch (error) {
        setStatus(`Error loading Frame B: ${error.message}`, true);
    }
});

// 窗口大小调整时重新计算图片显示
let resizeTimeout;
window.addEventListener('resize', () => {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
        if (state.currentImageA) {
            loadFrameImage(elements.frameASelect.value, elements.canvasA).then(img => {
                state.currentImageA = img;
                const ctx = elements.canvasA.getContext('2d');
                ctx.clearRect(0, 0, elements.canvasA.width, elements.canvasA.height);
                ctx.drawImage(img, 0, 0);
                // 重绘选区
                if (state.roi.width > 0 && state.roi.height > 0) {
                    ctx.strokeStyle = '#0e639c';
                    ctx.lineWidth = 2;
                    ctx.strokeRect(state.roi.x, state.roi.y, state.roi.width, state.roi.height);
                    ctx.fillStyle = 'rgba(14, 99, 156, 0.2)';
                    ctx.fillRect(state.roi.x, state.roi.y, state.roi.width, state.roi.height);
                }
            });
        }
        if (state.currentImageB) {
            loadFrameImage(elements.frameBSelect.value, elements.canvasB).then(img => {
                state.currentImageB = img;
                // 同步绘制选区
                drawSelectionOnFrameB();
            });
        }
    }, 250);
});

// 初始化
setupCanvasSelection();
loadFrames();
