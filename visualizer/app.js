// Initialize the cytoscape dagre extension
cytoscape.use(cytoscapeDagre);

// State variables
let numNodes = 0;           // n
let numGraphs = 0;          // d
let currentGraphIndex = 0;
let currentThreshold = 0.1; // Edge Threshold
let currentBaseScoreThreshold = 0; // Node Base Score Threshold
let cy = null;
let currentHorizontalSpacing = 30; // nodeSep in dagre
let currentVerticalSpacing = 80;   // rankSep in dagre
let showSupports = true;
let showAttacks = true;
let useBorderBaseScore = true; // Toggle for border thickness based on base score
let useTransparentFinalStrength = true; // Toggle for node opacity based on final strength
let removeSpikes = false; // Toggle to remove spikes (nodes not connected to default cases)
let hideDefaultsPerClass = false; // Toggle to hide defaults if their class is hidden
let currentFinalStrengthThreshold = 0.0;
let currentMinFStrength = 0.0;
let currentMaxFStrength = 1.0;
let showImages = false;     // Toggle to render images
let hasImages = false;      // True if X_train is 3D/4D
let isTabularData = false;  // True if X_train is Tabular

let visibleClasses = new Set();
let showDefaultCases = true;

let precomputedEdges = [];  // Array of arrays storing edges per graph
let yTrainData = [];
let defaultIndexesData = [];
let baseScoresData = [];    // Base scores (n, d)
let xTrainData = [];        // Raw feature data
let nodeImages = {};        // Map of nodeId -> Base64 Image URL
let normalizationMean = null;
let normalizationStd = null;

// New Cases Data
let newCasesData = [];
let newCasesLabels = [];
let newCasesBaseScores = [];
let newCasesAdjacency = [];
let finalStrengths = []; // Final strengths when a new case is active
let newCaseImages = {}; // Map of newCaseIndex -> Base64 Image URL
let selectedNewCaseIndex = -1;

// G-RAE Perturbation Data
// Legacy (single-target) shape: casebase_edges (B,n,n,d), new_case_edges (B,n,d),
// target_indices (B,) -- one target default per sample.
// Multi-target shape: casebase_edges (B,T,n,n,d), new_case_edges (B,T,n,d),
// target_indices (B,T) -- several candidate target defaults per sample, so a
// single perturbed edge's effect on every class can be compared at once.
let graeCasebaseEdges = [];
let graeNewCaseEdges = [];
let graeTargetIndices = [];
let graeTargetLabels = []; // (B,T) optional class label per target column
let graeMultiTarget = false;
let selectedPerturbationEdge = null; // { type: 'casebase', i, j } | { type: 'new_case', j }
let currentGraeDelta = 0;

// UI Elements
const jsonUpload = document.getElementById('json-upload');
const graphSlider = document.getElementById('graph-slider');
const thresholdSlider = document.getElementById('threshold-slider');
const baseScoreSlider = document.getElementById('base-score-slider');
const hSpacingSlider = document.getElementById('h-spacing-slider');
const vSpacingSlider = document.getElementById('v-spacing-slider');
const hSpacingLabel = document.getElementById('h-spacing-label');
const vSpacingLabel = document.getElementById('v-spacing-label');
const supportsCheckbox = document.getElementById('supports-checkbox');
const attacksCheckbox = document.getElementById('attacks-checkbox');
const borderBaseScoreCheckbox = document.getElementById('border-base-score-checkbox');
const showImagesCheckbox = document.getElementById('show-images-checkbox');
const imageToggleContainer = document.getElementById('image-toggle-container');
const transparentFinalStrengthCheckbox = document.getElementById('transparent-final-strength-checkbox');
const showSupportsCheckbox = document.getElementById('supports-checkbox');
const showAttacksCheckbox = document.getElementById('attacks-checkbox');
const removeSpikesCheckbox = document.getElementById('remove-spikes-checkbox');
const loadingOverlay = document.getElementById('loading-overlay');
const legendContainer = document.getElementById('legend-container');
const legendItems = document.getElementById('legend-items');

// New toggle controls
const sidebarToggleBtn = document.getElementById('sidebar-toggle');
const sidebar = document.getElementById('sidebar');
const legendHeader = document.getElementById('legend-header');
const legendIcon = document.getElementById('legend-icon');

// New Cases UI
const btnOpenNewCases = document.getElementById('btn-open-new-cases');
const newCasesList = document.getElementById('new-cases-list');
const finalStrengthControls = document.getElementById('final-strength-controls');
const finalStrengthSlider = document.getElementById('final-strength-slider');
const finalStrengthLabel = document.getElementById('final-strength-label');

// G-RAE Perturbation UI
const graePanel = document.getElementById('grae-panel');
const graeNoSelection = document.getElementById('grae-no-selection');
const graeSelection = document.getElementById('grae-selection');
const graeEdgeLabel = document.getElementById('grae-edge-label');
const graeDeltaSlider = document.getElementById('grae-delta-slider');
const graeDeltaLabel = document.getElementById('grae-delta-label');
const graeOriginalStrengthEl = document.getElementById('grae-original-strength');
const graePredictedStrengthEl = document.getElementById('grae-predicted-strength');
const graeClearBtn = document.getElementById('grae-clear-btn');
const graeSingleTarget = document.getElementById('grae-single-target');
const graeMultiTargetEl = document.getElementById('grae-multi-target');
const graeCompetitionTable = document.getElementById('grae-competition-table').querySelector('tbody');
const graeFlipHint = document.getElementById('grae-flip-hint');

const graphLabel = document.getElementById('graph-label');
const thresholdLabel = document.getElementById('threshold-label');
const baseScoreLabel = document.getElementById('base-score-label');

const loadingText = document.getElementById('loading-text');
const nodeTooltip = document.getElementById('node-tooltip');

const sidebarOpenBtn = document.getElementById('sidebar-open');

let debounceTimeout = null;

sidebarToggleBtn.addEventListener('click', () => {
    sidebar.classList.add('collapsed');
    sidebarOpenBtn.style.display = 'block';
});

sidebarOpenBtn.addEventListener('click', () => {
    sidebar.classList.remove('collapsed');
    sidebarOpenBtn.style.display = 'none';
});

legendHeader.addEventListener('click', () => {
    if (legendItems.style.display === 'none') {
        legendItems.style.display = 'flex';
        legendIcon.textContent = '▼';
    } else {
        legendItems.style.display = 'none';
        legendIcon.textContent = '▶';
    }
});

function showLoading(text = "Processing...") {
    loadingText.textContent = text;
    loadingOverlay.classList.remove('hidden');
}

function hideLoading() {
    loadingOverlay.classList.add('hidden');
}

/**
 * Maps a weight [-1, 1] to an rgb color.
 */
function getColor(weight) {
    const w = Math.max(-1, Math.min(1, weight)); 
    const absW = Math.abs(w);
    const intensity = Math.round(absW * 255);
    
    if (w > 0) {
        const r = 255 - intensity;
        const g = 255 - Math.round(intensity * (105 / 255));
        const b = 255 - intensity;
        return `rgb(${r}, ${g}, ${b})`;
    } else if (w < 0) {
        const r = 255;
        const g = 255 - intensity;
        const b = 255 - intensity;
        return `rgb(${r}, ${g}, ${b})`;
    }
    
    return 'rgb(255,255,255)';
}

/**
 * Maps negative attack weights to a purple gradient.
 */
function getPurpleColor(weight) {
    // Only used for negative weights (attacks)
    const w = Math.max(-1, Math.min(0, weight));
    const absW = Math.abs(w);
    // Dark purple at -1 (e.g., 128, 0, 128), white at 0
    const intensity = 1 - absW; // 0 at max attack, 1 at no attack
    const r = Math.round(128 + (127 * intensity));
    const g = Math.round(0 + (255 * intensity));
    const b = Math.round(128 + (127 * intensity));
    return `rgb(${r}, ${g}, ${b})`;
}

/**
 * Generates an HSL color given an index and total number of classes.
 */
function getClassColor(index, totalClasses) {
    // Distribute hue evenly around 360 degrees
    const hue = Math.round((360 / totalClasses) * index);
    return `hsl(${hue}, 70%, 55%)`;
}

/**
 * Builds the legend in the sidebar
 */
function renderLegend(uniqueClasses) {
    legendItems.innerHTML = ''; // Clear existing
    legendContainer.style.display = 'block'; // Show container

    uniqueClasses.forEach((classInt, i) => {
        const color = getClassColor(i, uniqueClasses.length);
        
        const item = document.createElement('label');
        item.className = 'legend-item checkbox-row';
        item.style.cursor = 'pointer';
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = true;
        checkbox.addEventListener('change', (e) => {
            if (e.target.checked) {
                visibleClasses.add(classInt);
            } else {
                visibleClasses.delete(classInt);
            }
            if (precomputedEdges.length > 0) handleSliderChange();
        });

        const colorBox = document.createElement('span');
        colorBox.className = 'color-box';
        colorBox.style.backgroundColor = color;
        
        const labelText = document.createTextNode(`Class ${classInt}`);
        
        item.appendChild(checkbox);
        item.appendChild(colorBox);
        item.appendChild(labelText);
        legendItems.appendChild(item);
    });

    // Add Default Case Legend Item
    if (defaultIndexesData && defaultIndexesData.length > 0) {
        const defaultItem = document.createElement('label');
        defaultItem.className = 'legend-item checkbox-row';
        defaultItem.style.cursor = 'pointer';
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = true;
        checkbox.addEventListener('change', (e) => {
            showDefaultCases = e.target.checked;
            if (precomputedEdges.length > 0) handleSliderChange();
        });

        const defaultBox = document.createElement('span');
        defaultBox.className = 'default-box';
        
        const defaultLabel = document.createTextNode(`Default Case`);
        
        defaultItem.appendChild(checkbox);
        defaultItem.appendChild(defaultBox);
        defaultItem.appendChild(defaultLabel);
        legendItems.appendChild(defaultItem);

        const hideDefaultsItem = document.createElement('label');
        hideDefaultsItem.className = 'legend-item checkbox-row';
        hideDefaultsItem.style.cursor = 'pointer';
        
        const hideDefaultsCheckbox = document.createElement('input');
        hideDefaultsCheckbox.type = 'checkbox';
        hideDefaultsCheckbox.checked = hideDefaultsPerClass;
        hideDefaultsCheckbox.addEventListener('change', (e) => {
            hideDefaultsPerClass = e.target.checked;
            if (precomputedEdges.length > 0) handleSliderChange();
        });

        const hideDefaultsLabel = document.createTextNode(`Hide Defaults Per Class`);
        
        hideDefaultsItem.appendChild(hideDefaultsCheckbox);
        hideDefaultsItem.appendChild(hideDefaultsLabel);
        legendItems.appendChild(hideDefaultsItem);
    }
}

/**
 * Pre-processes the massive matrix once on upload.

 * Builds an array of edges for each dimension 'd' so we don't have to loop n*n repeatedly.
 */
function precomputeEdges(matrixData) {
    numNodes = matrixData.length;
    numGraphs = matrixData[0][0].length;
    precomputedEdges = Array.from({ length: numGraphs }, () => []);

    for (let i = 0; i < numNodes; i++) {
        for (let j = 0; j < numNodes; j++) {
            for (let d = 0; d < numGraphs; d++) {
                const weight = matrixData[i][j][d];
                if (weight !== 0) {
                    precomputedEdges[d].push({
                        data: {
                            id: `e_${i}_${j}_${d}`,
                            source: `n${i}`,
                            target: `n${j}`,
                            weight: weight,
                            lineColor: getColor(weight),
                            width: Math.max(1, Math.abs(weight) * 5)
                        }
                    });
                }
            }
        }
    }
}

/**
 * Detects if X_train contains images or tabular data and precomputes Base64 URIs.
 * Expects shape: (n, H, W) or (n, C, H, W) or (n, H, W, C) for images
 * Expects shape: (n, features) for tabular data
 */
function precomputeImages() {
    hasImages = false;
    nodeImages = {};
    newCaseImages = {};
    if (!xTrainData || xTrainData.length === 0) return;

    const sample = xTrainData[0];
    
    // Check array depth to infer shape
    let isImage = false;
    let isTabular = false;
    let isGrayscale = true;
    let H = 0, W = 0;

    if (Array.isArray(sample)) {
        if (typeof sample[0] === 'number') {
            isTabular = true;
        } else if (Array.isArray(sample[0])) {
            if (!Array.isArray(sample[0][0])) {
                // Shape: (H, W) -> Grayscale
                isImage = true;
                H = sample.length;
                W = sample[0].length;
            } else if (Array.isArray(sample[0][0])) {
                // Shape: (C, H, W) or (H, W, C)
                isImage = true;
                if (sample.length === 1 || sample.length === 3) {
                    // (C, H, W)
                    isGrayscale = sample.length === 1;
                    H = sample[0].length;
                    W = sample[0][0].length;
                } else {
                    // (H, W, C)
                    isGrayscale = sample[0][0].length === 1;
                    H = sample.length;
                    W = sample[0].length;
                }
            }
        }
    }

    if (!isImage && !isTabular) return;

    hasImages = true;
    isTabularData = isTabular;
    imageToggleContainer.style.display = 'flex';
    showImagesCheckbox.disabled = false;
    showImagesCheckbox.checked = false;
    
    const labelElem = document.querySelector('label[for="show-images-checkbox"]');
    if (labelElem) {
        labelElem.textContent = isTabular ? "Show Features Bar Chart" : "Show Node Images";
    }

    if (isTabular) {
        function renderTabularToDataUrl(item) {
            const canvas = document.createElement('canvas');
            const width = 800;
            const height = 200;
            canvas.width = width;
            canvas.height = height;
            const ctx = canvas.getContext('2d');
            
            // Draw background
            ctx.fillStyle = '#f8f9fa';
            ctx.fillRect(0, 0, width, height);
            
            // Find min/max for scaling
            let minVal = Math.min(...item);
            let maxVal = Math.max(...item);
            if (minVal === maxVal) {
                minVal -= 1;
                maxVal += 1;
            }
            const range = maxVal - minVal;
            
            const padding = 10;
            const drawWidth = width - 2 * padding;
            const drawHeight = height - 2 * padding;
            
            const numFeatures = item.length;
            const barWidth = Math.max(1, (drawWidth / numFeatures) - 1);
            
            // Draw axes
            ctx.strokeStyle = '#333';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(padding, height - padding);
            ctx.lineTo(width - padding, height - padding); // X axis
            const zeroY = height - padding - ((0 - minVal) / range) * drawHeight;
            if (minVal < 0 && maxVal > 0) {
                ctx.moveTo(padding, zeroY);
                ctx.lineTo(width - padding, zeroY); // Zero line
                ctx.strokeStyle = '#aaa';
            }
            ctx.stroke();
            
            // Draw bars
            for (let i = 0; i < numFeatures; i++) {
                const val = item[i];
                const normalizedH = (Math.abs(val) / range) * drawHeight;
                const x = padding + i * (drawWidth / numFeatures);
                let y;
                if (val >= 0) {
                    y = (minVal < 0 && maxVal > 0) ? zeroY - normalizedH : height - padding - normalizedH;
                    ctx.fillStyle = '#3498db'; // blue
                } else {
                    y = zeroY;
                    ctx.fillStyle = '#e74c3c'; // red
                }
                
                ctx.fillRect(x, y, barWidth, Math.max(1, normalizedH));
            }
            
            // Add border
            ctx.strokeStyle = '#ccc';
            ctx.strokeRect(0, 0, width, height);
            
            return canvas.toDataURL('image/png');
        }

        for (let n = 0; n < xTrainData.length; n++) {
            nodeImages[n] = renderTabularToDataUrl(xTrainData[n]);
        }

        if (newCasesData && newCasesData.length > 0) {
            for (let i = 0; i < newCasesData.length; i++) {
                newCaseImages[i] = renderTabularToDataUrl(newCasesData[i]);
            }
        }
        
    } else {
        // Create an off-screen canvas for rendering
        // Scale up small images (like 32x32 CIFAR) directly on the canvas using nearest-neighbor
        // to avoid browser anti-aliasing blur when Cytoscape draws them.
        const SCALE = 2; 
        const canvas = document.createElement('canvas');
        canvas.width = W * SCALE;
        canvas.height = H * SCALE;
        const ctx = canvas.getContext('2d');
        const imgData = ctx.createImageData(canvas.width, canvas.height);

        function renderItemToDataUrl(item) {
            // Use dynamically provided normalization parameters or fallback to CIFAR-10 defaults
            // X = (X - mean) / std  -->  Original = (X * std) + mean
            let means = [0.4914, 0.4822, 0.4465];
            let stds = [0.2470, 0.2435, 0.2616];

            if (normalizationMean && normalizationStd) {
                means = Array.isArray(normalizationMean) ? normalizationMean.flat() : [normalizationMean];
                stds = Array.isArray(normalizationStd) ? normalizationStd.flat() : [normalizationStd];
            }
            
            // Flatten and un-normalize values
            const unnormalizedVals = []; // will store objects {r,g,b} or just single values
            
            if (H > 0 && W > 0 && !Array.isArray(item[0][0])) {
                // (H, W) -> Grayscale, just map back assuming typical 0-1 or we use first channel mean/std
                for(let r=0; r<H; r++){
                    for(let c=0; c<W; c++){
                        const v = (item[r][c] * stds[0]) + means[0]; 
                        unnormalizedVals.push({ r: v, g: v, b: v });
                    }
                }
            } else if (item.length === 1 || item.length === 3) {
                // (C, H, W)
                const numChannels = item.length;
                for(let r=0; r<H; r++){
                    for(let w=0; w<W; w++){
                        let rv = 0, gv = 0, bv = 0;
                        if (numChannels === 3) {
                            rv = (item[0][r][w] * stds[0]) + means[0];
                            gv = (item[1][r][w] * stds[1]) + means[1];
                            bv = (item[2][r][w] * stds[2]) + means[2];
                        } else {
                            const v = (item[0][r][w] * stds[0]) + means[0];
                            rv = v; gv = v; bv = v;
                        }
                        unnormalizedVals.push({ r: rv, g: gv, b: bv });
                    }
                }
            } else {
                // (H, W, C)
                const numChannels = item[0][0].length;
                for(let r=0; r<H; r++){
                    for(let w=0; w<W; w++){
                        let rv = 0, gv = 0, bv = 0;
                        if (numChannels === 3) {
                            rv = (item[r][w][0] * stds[0]) + means[0];
                            gv = (item[r][w][1] * stds[1]) + means[1];
                            bv = (item[r][w][2] * stds[2]) + means[2];
                        } else {
                            const v = (item[r][w][0] * stds[0]) + means[0];
                            rv = v; gv = v; bv = v;
                        }
                        unnormalizedVals.push({ r: rv, g: gv, b: bv });
                    }
                }
            }

            // Fill ImageData (apply manual nearest-neighbor scaling)
            for (let i = 0; i < unnormalizedVals.length; i++) {
                const pixel = unnormalizedVals[i];
                
                // Map from [0, 1] range to [0, 255] and clamp
                const rScaled = Math.max(0, Math.min(255, Math.round(pixel.r * 255)));
                const gScaled = Math.max(0, Math.min(255, Math.round(pixel.g * 255)));
                const bScaled = Math.max(0, Math.min(255, Math.round(pixel.b * 255)));

                // Calculate original row/col
                const origR = Math.floor(i / W);
                const origC = i % W;

                // Fill a SCALE x SCALE block of pixels in the scaled image
                for (let sr = 0; sr < SCALE; sr++) {
                    for (let sc = 0; sc < SCALE; sc++) {
                        const destR = (origR * SCALE) + sr;
                        const destC = (origC * SCALE) + sc;
                        
                        const destIdx = (destR * (W * SCALE) + destC) * 4;
                        
                        imgData.data[destIdx] = rScaled;    // R
                        imgData.data[destIdx+1] = gScaled;  // G
                        imgData.data[destIdx+2] = bScaled;  // B
                        imgData.data[destIdx+3] = 255;      // Alpha
                    }
                }
            }

            ctx.putImageData(imgData, 0, 0);
            return canvas.toDataURL('image/png');
        }

        for (let n = 0; n < xTrainData.length; n++) {
            nodeImages[n] = renderItemToDataUrl(xTrainData[n]);
        }

        if (newCasesData && newCasesData.length > 0) {
            for (let i = 0; i < newCasesData.length; i++) {
                newCaseImages[i] = renderItemToDataUrl(newCasesData[i]);
            }
        }
    }
}

/**
 * Populates the New Cases UI list
 */
function renderNewCasesList() {
    newCasesList.innerHTML = '';
    
    if (!newCasesData || newCasesData.length === 0) {
        btnOpenNewCases.style.display = 'none';
        return;
    }
    
    btnOpenNewCases.style.display = 'block';
    const uniqueClasses = [...new Set(yTrainData)].sort((a, b) => a - b);
    
    newCasesData.forEach((_, i) => {
        const itemDiv = document.createElement('div');
        itemDiv.className = 'new-case-item';
        if (selectedNewCaseIndex === i) {
            itemDiv.classList.add('selected');
        }
        
        const infoContainer = document.createElement('div');
        infoContainer.className = 'new-case-info-container';
        
        if (hasImages && newCaseImages[i]) {
            const img = document.createElement('img');
            img.src = newCaseImages[i];
            img.className = 'new-case-image';
            infoContainer.appendChild(img);
        }

        const headerDiv = document.createElement('div');
        headerDiv.className = 'new-case-header';
        
        const titleSpan = document.createElement('span');
        titleSpan.className = 'new-case-title';
        titleSpan.textContent = `Case N_${i}`;
        
        const classSpan = document.createElement('span');
        classSpan.className = 'new-case-class';
        const classInt = parseInt(newCasesLabels[i]);
        if (!isNaN(classInt)) {
            const classIndex = uniqueClasses.indexOf(classInt);
            const color = getClassColor(classIndex, uniqueClasses.length);
            
            const colorBox = document.createElement('span');
            colorBox.className = 'color-box';
            colorBox.style.backgroundColor = color;
            colorBox.style.width = '12px';
            colorBox.style.height = '12px';
            
            classSpan.appendChild(colorBox);
            classSpan.appendChild(document.createTextNode(`Class ${classInt}`));
        } else {
            classSpan.textContent = 'Unknown Class';
        }
        
        headerDiv.appendChild(titleSpan);
        headerDiv.appendChild(classSpan);
        infoContainer.appendChild(headerDiv);
        itemDiv.appendChild(infoContainer);
        
        const button = document.createElement('button');
        if (selectedNewCaseIndex === i) {
            button.textContent = 'Remove from Graph';
            button.className = 'btn btn-sm btn-danger';
        } else {
            button.textContent = 'Add to Graph';
            button.className = 'btn btn-sm btn-primary';
        }
        
        button.onclick = () => {
            if (selectedNewCaseIndex === i) {
                selectedNewCaseIndex = -1;
                finalStrengthControls.style.display = 'none';
            } else {
                selectedNewCaseIndex = i;
                if (finalStrengths && finalStrengths.length > 0) {
                    finalStrengthControls.style.display = 'block';
                    updateFStrengthBounds();
                }
            }
            selectedPerturbationEdge = null;
            currentGraeDelta = 0;
            graeDeltaSlider.value = 0;
            updateGraePanel();
            renderNewCasesList(); // re-render to update classes/buttons
            if (precomputedEdges.length > 0) handleSliderChange();
            
            // Auto close the modal
            const modalEl = document.getElementById('newCasesModal');
            if (modalEl) {
                const modalInstance = bootstrap.Modal.getInstance(modalEl);
                if (modalInstance) {
                    modalInstance.hide();
                }
            }
        };
        
        itemDiv.appendChild(button);
        newCasesList.appendChild(itemDiv);
    });
}

/**
 * Handle file upload
 */
jsonUpload.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (!file) return;

    showLoading("Parsing JSON...");

    const reader = new FileReader();
    reader.onload = (e) => {
        // Yield to the browser to render the loading screen
        setTimeout(() => {
            try {
                const data = JSON.parse(e.target.result);
                if (data.adjacency_matrix) {
                    showLoading("Precomputing edges...");
                    
                    // Store new data
                    yTrainData = data.y_train || [];
                    defaultIndexesData = data.default_indexes || [];
                    baseScoresData = data.base_scores || [];
                    xTrainData = data.X_train || [];
                    normalizationMean = data.normalization_mean || null;
                    normalizationStd = data.normalization_std || null;
                    
                    newCasesData = data.new_cases || [];
                    newCasesLabels = data.new_cases_labels || [];
                    newCasesBaseScores = data.new_cases_base_scores || [];
                    newCasesAdjacency = data.new_cases_adjacency || [];
                    finalStrengths = data.final_strengths || [];
                    selectedNewCaseIndex = -1;

                    graeCasebaseEdges = (data.grae && data.grae.casebase_edges) || [];
                    graeNewCaseEdges = (data.grae && data.grae.new_case_edges) || [];
                    graeTargetIndices = (data.grae && data.grae.target_indices) || [];
                    graeTargetLabels = (data.grae && data.grae.target_labels) || [];
                    graeMultiTarget = Array.isArray(graeTargetIndices[0]);
                    selectedPerturbationEdge = null;
                    currentGraeDelta = 0;

                    // Initialize filter states
                    visibleClasses = new Set(yTrainData);
                    showDefaultCases = true;
                    hideDefaultsPerClass = false;

                    setTimeout(() => {
                      try {
                        precomputeImages();
                        precomputeEdges(data.adjacency_matrix);

                        // Handle Colors and Legend
                        let uniqueClasses = [];
                        if (yTrainData.length > 0) {
                            uniqueClasses = [...new Set(yTrainData)].sort((a, b) => a - b);
                            renderLegend(uniqueClasses);
                        } else {
                            legendContainer.style.display = 'none';
                        }
                        
                        renderNewCasesList();
                        
                        // Reset states
                        currentGraphIndex = 0;
                        currentThreshold = 0.1;
                        currentBaseScoreThreshold = 0.0;
                        currentFinalStrengthThreshold = 0.0;
                        currentMinFStrength = 0.0;
                        currentMaxFStrength = 1.0;
                        currentHorizontalSpacing = 30;
                        currentVerticalSpacing = 80;
                        showSupports = true;
                        showAttacks = true;
                        useBorderBaseScore = true;
                        useTransparentFinalStrength = true;
                        showImages = false;
                        removeSpikes = false;
                        
                        // Handle Graph Slider Visibility
                        const graphSliderContainer = document.getElementById('graph-slider-container');
                        if (numGraphs > 1) {
                            graphSliderContainer.style.display = 'block';
                            graphSlider.min = 0;
                            graphSlider.max = numGraphs - 1;
                            graphSlider.value = 0;
                            graphSlider.disabled = false;
                        } else {
                            graphSliderContainer.style.display = 'none';
                        }
                        
                        thresholdSlider.value = 0.1;
                        thresholdSlider.disabled = false;

                        baseScoreSlider.value = 0.0;
                        baseScoreSlider.disabled = false;

                        finalStrengthSlider.min = 0.0;
                        finalStrengthSlider.max = 1.0;
                        finalStrengthSlider.step = 0.05;
                        finalStrengthSlider.value = 0.0;
                        
                        borderBaseScoreCheckbox.checked = true;
                        borderBaseScoreCheckbox.disabled = false;
                        
                        transparentFinalStrengthCheckbox.checked = true;

                        hSpacingSlider.value = 30;
                        hSpacingSlider.disabled = false;

                        vSpacingSlider.value = 80;
                        vSpacingSlider.disabled = false;

                        supportsCheckbox.checked = true;
                        supportsCheckbox.disabled = false;

                        attacksCheckbox.checked = true;
                        attacksCheckbox.disabled = false;
                        
                        removeSpikesCheckbox.checked = false;
                        removeSpikesCheckbox.disabled = false;

                        graeDeltaSlider.value = 0;
                        updateGraePanel();

                        updateLabels();
                        showLoading("Rendering graph...");

                        setTimeout(() => {
                            try {
                                initCytoscape();
                            } catch (err) {
                                console.error("Error rendering graph:", err);
                                alert(`Error rendering graph: ${err.message}`);
                            } finally {
                                hideLoading();
                            }
                        }, 50);
                      } catch (err) {
                          console.error("Error precomputing graph data:", err);
                          alert(`Error precomputing graph data: ${err.message}`);
                          hideLoading();
                      }
                    }, 50);
                } else {
                    alert("Invalid JSON format. Must contain 'adjacency_matrix'.");
                    hideLoading();
                }
            } catch (err) {
                console.error("Error parsing JSON:", err);
                alert("Error parsing JSON file.");
                hideLoading();
            }
        }, 50);
    };
    reader.readAsText(file);
});

/**
 * Event Listeners with Debouncing
 */
function handleSliderChange() {
    clearTimeout(debounceTimeout);
    debounceTimeout = setTimeout(() => {
        showLoading("Updating graph...");
        setTimeout(() => {
            updateGraphElements();
            hideLoading();
        }, 10);
    }, 150); // 150ms debounce
}

graphSlider.addEventListener('input', (e) => {
    currentGraphIndex = parseInt(e.target.value, 10);
    updateFStrengthBounds();
    updateLabels();
    updateGraePanel();
    if (precomputedEdges.length > 0) handleSliderChange();
});

thresholdSlider.addEventListener('input', (e) => {
    currentThreshold = parseFloat(e.target.value);
    updateLabels();
    if (precomputedEdges.length > 0) handleSliderChange();
});

baseScoreSlider.addEventListener('input', (e) => {
    currentBaseScoreThreshold = parseFloat(e.target.value);
    updateLabels();
    if (precomputedEdges.length > 0) handleSliderChange();
});

finalStrengthSlider.addEventListener('input', (e) => {
    currentFinalStrengthThreshold = parseFloat(e.target.value);
    updateLabels();
    if (precomputedEdges.length > 0) handleSliderChange();
});

function handleSpacingChange() {
    clearTimeout(debounceTimeout);
    debounceTimeout = setTimeout(() => {
        showLoading("Updating layout...");
        setTimeout(() => {
            runLayout();
            hideLoading();
        }, 10);
    }, 150);
}

hSpacingSlider.addEventListener('input', (e) => {
    currentHorizontalSpacing = parseInt(e.target.value, 10);
    updateLabels();
    if (precomputedEdges.length > 0) handleSpacingChange();
});

vSpacingSlider.addEventListener('input', (e) => {
    currentVerticalSpacing = parseInt(e.target.value, 10);
    updateLabels();
    if (precomputedEdges.length > 0) handleSpacingChange();
});

supportsCheckbox.addEventListener('change', (e) => {
    showSupports = e.target.checked;
    if (precomputedEdges.length > 0) handleSliderChange();
});

attacksCheckbox.addEventListener('change', (e) => {
    showAttacks = e.target.checked;
    if (precomputedEdges.length > 0) handleSliderChange();
});

borderBaseScoreCheckbox.addEventListener('change', (e) => {
    useBorderBaseScore = e.target.checked;
    if (precomputedEdges.length > 0) handleSliderChange();
});

transparentFinalStrengthCheckbox.addEventListener('change', (e) => {
    useTransparentFinalStrength = e.target.checked;
    if (precomputedEdges.length > 0) handleSliderChange();
});

removeSpikesCheckbox.addEventListener('change', (e) => {
    removeSpikes = e.target.checked;
    if (precomputedEdges.length > 0) handleSliderChange();
});

showImagesCheckbox.addEventListener('change', (e) => {
    showImages = e.target.checked;
    if (precomputedEdges.length > 0) handleSliderChange();
});

function updateLabels() {
    graphLabel.textContent = `Graph: ${currentGraphIndex} / ${Math.max(0, numGraphs - 1)}`;
    thresholdLabel.textContent = `Edge Threshold: ${currentThreshold.toFixed(2)}`;
    baseScoreLabel.textContent = `Base Score \u2265 ${currentBaseScoreThreshold.toFixed(2)}`;
    finalStrengthLabel.textContent = `Final Strength \u2265 ${currentFinalStrengthThreshold.toFixed(2)}`;
    hSpacingLabel.textContent = `Horizontal Spacing: ${currentHorizontalSpacing}`;
    vSpacingLabel.textContent = `Vertical Spacing: ${currentVerticalSpacing}`;
}

/**
 * Filters the precomputed edges based on threshold and edge type
 */
function updateFStrengthBounds() {
    if (selectedNewCaseIndex !== -1 && finalStrengths && finalStrengths.length > 0) {
        currentMinFStrength = Infinity;
        currentMaxFStrength = -Infinity;
        for (let i = 0; i < numNodes; i++) {
            let fs = getFinalStrength(selectedNewCaseIndex, i, currentGraphIndex);
            if (fs < currentMinFStrength) currentMinFStrength = fs;
            if (fs > currentMaxFStrength) currentMaxFStrength = fs;
        }
        if (currentMinFStrength === Infinity) {
            currentMinFStrength = 0;
            currentMaxFStrength = 1;
        } else if (currentMaxFStrength === currentMinFStrength) {
            currentMaxFStrength = currentMinFStrength + 1; // avoid division by zero
        }
        
        finalStrengthSlider.min = currentMinFStrength;
        finalStrengthSlider.max = currentMaxFStrength;
        finalStrengthSlider.step = (currentMaxFStrength - currentMinFStrength) / 100;
        
        if (currentFinalStrengthThreshold < currentMinFStrength) currentFinalStrengthThreshold = currentMinFStrength;
        if (currentFinalStrengthThreshold > currentMaxFStrength) currentFinalStrengthThreshold = currentMaxFStrength;
        finalStrengthSlider.value = currentFinalStrengthThreshold;
    }
}

/**
 * Looks up the G-RAE gradient (d(target strength)/d(edge weight)) for the
 * currently selected perturbation edge, for every candidate target default
 * of the active new case (just one, in the common single-target export;
 * several, when the export was built with multiple targets per sample so
 * competing classes can be compared side by side).
 */
function getSelectedGraeTargets() {
    if (!selectedPerturbationEdge || selectedNewCaseIndex === -1) return null;
    const caseIdx = selectedNewCaseIndex;
    const rawTargets = graeTargetIndices[caseIdx];
    if (rawTargets === undefined) return null;

    const d = currentGraphIndex;
    const targetDefaultIdxList = graeMultiTarget ? rawTargets : [rawTargets];
    const labelsForCase = graeMultiTarget ? (graeTargetLabels[caseIdx] || []) : [];

    let edgeLabelText = '';
    const targets = targetDefaultIdxList.map((targetDefaultIdx, t) => {
        let gradient = 0;

        if (selectedPerturbationEdge.type === 'casebase') {
            const { i, j } = selectedPerturbationEdge;
            const sampleEdges = graeMultiTarget ? graeCasebaseEdges[caseIdx] && graeCasebaseEdges[caseIdx][t] : graeCasebaseEdges[caseIdx];
            const row = sampleEdges && sampleEdges[i] && sampleEdges[i][j];
            gradient = row ? row[d] : 0;
            edgeLabelText = `Casebase edge: n${i} → n${j}`;
        } else {
            const { j } = selectedPerturbationEdge;
            const sampleEdges = graeMultiTarget ? graeNewCaseEdges[caseIdx] && graeNewCaseEdges[caseIdx][t] : graeNewCaseEdges[caseIdx];
            const row = sampleEdges && sampleEdges[j];
            gradient = row ? row[d] : 0;
            edgeLabelText = `New Case ${caseIdx} → n${j}`;
        }

        const targetNodeId = defaultIndexesData[targetDefaultIdx];
        const originalStrength = getFinalStrength(caseIdx, targetNodeId, d);
        const classLabel = labelsForCase[t];
        return { targetDefaultIdx, targetNodeId, gradient, originalStrength, classLabel };
    });

    return { edgeLabelText, targets };
}

/**
 * Whether the currently selected new case has any G-RAE data to perturb.
 */
function graeAvailableForCurrentCase() {
    return selectedNewCaseIndex !== -1 && graeTargetIndices[selectedNewCaseIndex] !== undefined;
}

/**
 * Refreshes the G-RAE sidebar panel and the live linear-approximation
 * readout, and pulses the differentiated default node so the effect of the
 * perturbation is visible directly on the graph.
 */
function updateGraePanel() {
    if (!graeAvailableForCurrentCase()) {
        graePanel.style.display = 'none';
        clearGraeNodeHighlight();
        return;
    }

    graePanel.style.display = 'block';

    if (!selectedPerturbationEdge) {
        graeNoSelection.style.display = 'block';
        graeSelection.style.display = 'none';
        clearGraeNodeHighlight();
        return;
    }

    const info = getSelectedGraeTargets();
    if (!info) {
        selectedPerturbationEdge = null;
        graeNoSelection.style.display = 'block';
        graeSelection.style.display = 'none';
        clearGraeNodeHighlight();
        return;
    }

    graeNoSelection.style.display = 'none';
    graeSelection.style.display = 'block';

    graeEdgeLabel.textContent = info.edgeLabelText;
    graeDeltaLabel.textContent = `Δ = ${currentGraeDelta.toFixed(2)}`;

    if (graeMultiTarget && info.targets.length > 1) {
        graeSingleTarget.style.display = 'none';
        graeMultiTargetEl.style.display = 'block';
        renderGraeCompetitionTable(info.targets);
    } else {
        graeSingleTarget.style.display = 'block';
        graeMultiTargetEl.style.display = 'none';

        const t = info.targets[0];
        const predicted = t.originalStrength + t.gradient * currentGraeDelta;
        graeOriginalStrengthEl.textContent = t.originalStrength.toFixed(4);
        graePredictedStrengthEl.textContent = predicted.toFixed(4);
        highlightGraeTargetNode(t.targetNodeId, predicted);
    }
}

/**
 * Renders the per-class "competition" table for multi-target G-RAE data:
 * every candidate target's live predicted strength under the current Δ,
 * with the current winner bolded, plus a closed-form hint for the nearest
 * Δ (within the slider's [-1, 1] range) at which some other target would
 * overtake it -- since predicted strength is linear in Δ, the crossing
 * point of two targets' lines is exact, not searched for.
 */
function renderGraeCompetitionTable(targets) {
    const uniqueClasses = [...new Set(yTrainData)].sort((a, b) => a - b);
    const predicted = targets.map(t => t.originalStrength + t.gradient * currentGraeDelta);
    const winnerIdx = predicted.indexOf(Math.max(...predicted));

    graeCompetitionTable.innerHTML = '';
    targets.forEach((t, idx) => {
        const tr = document.createElement('tr');
        if (idx === winnerIdx) tr.className = 'fw-bold table-warning';

        const labelTd = document.createElement('td');
        if (t.classLabel !== undefined) {
            const classIndex = uniqueClasses.indexOf(t.classLabel);
            const color = getClassColor(classIndex, uniqueClasses.length);
            const colorBox = document.createElement('span');
            colorBox.className = 'color-box';
            colorBox.style.backgroundColor = color;
            colorBox.style.width = '10px';
            colorBox.style.height = '10px';
            colorBox.style.marginRight = '4px';
            labelTd.appendChild(colorBox);
            labelTd.appendChild(document.createTextNode(`Class ${t.classLabel}`));
        } else {
            labelTd.textContent = `Target ${idx}`;
        }

        const valTd = document.createElement('td');
        valTd.textContent = predicted[idx].toFixed(4);
        valTd.className = 'text-end';

        tr.appendChild(labelTd);
        tr.appendChild(valTd);
        graeCompetitionTable.appendChild(tr);
    });

    // Closed-form crossing: predicted[t](delta) = orig[t] + grad[t]*delta,
    // so predicted[t] == predicted[winner] at
    // delta* = (orig[winner] - orig[t]) / (grad[t] - grad[winner]).
    const winner = targets[winnerIdx];
    const hints = [];
    targets.forEach((t, idx) => {
        if (idx === winnerIdx) return;
        const gradDiff = t.gradient - winner.gradient;
        if (Math.abs(gradDiff) < 1e-9) return; // parallel lines never cross
        const deltaStar = (winner.originalStrength - t.originalStrength) / gradDiff;
        if (deltaStar >= -1 && deltaStar <= 1) {
            hints.push({ idx, deltaStar });
        }
    });

    if (hints.length > 0) {
        hints.sort((a, b) => Math.abs(a.deltaStar - currentGraeDelta) - Math.abs(b.deltaStar - currentGraeDelta));
        const nearest = hints[0];
        const label = targets[nearest.idx].classLabel !== undefined ? `Class ${targets[nearest.idx].classLabel}` : `Target ${nearest.idx}`;
        graeFlipHint.textContent = `⚡ ${label} overtakes at Δ ≈ ${nearest.deltaStar.toFixed(2)}`;
    } else {
        graeFlipHint.textContent = 'No flip within Δ ∈ [-1, 1] for this edge.';
    }

    highlightGraeTargetNode(targets[winnerIdx].targetNodeId, predicted[winnerIdx]);
}

/**
 * Temporarily overrides the differentiated default node's opacity/border to
 * reflect the linearly-predicted strength under the current perturbation.
 * This override is re-applied any time the panel refreshes, and cleared by
 * clearGraeNodeHighlight (e.g. on "Clear" or when the graph re-renders).
 */
function highlightGraeTargetNode(nodeId, predictedStrength) {
    if (!cy) return;
    const node = cy.getElementById(`n${nodeId}`);
    if (!node || node.length === 0) return;

    const clamped = Math.max(0, Math.min(1, predictedStrength));
    node.data('opacity', 0.10 + (0.90 * clamped));
    node.data('borderColor', '#e67e22');
    node.data('borderWidth', 6);
}

function clearGraeNodeHighlight() {
    if (precomputedEdges.length > 0) updateGraphElements();
}

graeDeltaSlider.addEventListener('input', (e) => {
    currentGraeDelta = parseFloat(e.target.value);
    updateGraePanel();
});

graeClearBtn.addEventListener('click', () => {
    selectedPerturbationEdge = null;
    currentGraeDelta = 0;
    graeDeltaSlider.value = 0;
    updateGraePanel();
});

function getFilteredEdges() {
    let edges = precomputedEdges[currentGraphIndex].filter(edge => {
        // Source/Target Base Score Visibility check
        const sourceId = parseInt(edge.data.source.substring(1));
        const targetId = parseInt(edge.data.target.substring(1));
        
        const isSourceDefault = defaultIndexesData.includes(sourceId);
        const isTargetDefault = defaultIndexesData.includes(targetId);
        
        const sourceScore = baseScoresData[sourceId] ? baseScoresData[sourceId][currentGraphIndex] : 1;
        const targetScore = baseScoresData[targetId] ? baseScoresData[targetId][currentGraphIndex] : 1;

        if (!isSourceDefault && sourceScore < currentBaseScoreThreshold) return false;
        if (!isTargetDefault && targetScore < currentBaseScoreThreshold) return false;

        if (selectedNewCaseIndex !== -1 && finalStrengths.length > 0) {
            const sourceFinalStrength = getFinalStrength(selectedNewCaseIndex, sourceId, currentGraphIndex);
            const targetFinalStrength = getFinalStrength(selectedNewCaseIndex, targetId, currentGraphIndex);
            if (!isSourceDefault && sourceFinalStrength < currentFinalStrengthThreshold) return false;
            if (!isTargetDefault && targetFinalStrength < currentFinalStrengthThreshold) return false;
        }

        // Class Filter Check
        const sourceClass = yTrainData[sourceId];
        const targetClass = yTrainData[targetId];
        
        if (isSourceDefault && !showDefaultCases) return false;
        if (isSourceDefault && hideDefaultsPerClass && sourceClass !== undefined && !visibleClasses.has(sourceClass)) return false;
        if (!isSourceDefault && sourceClass !== undefined && !visibleClasses.has(sourceClass)) return false;
        
        if (isTargetDefault && !showDefaultCases) return false;
        if (isTargetDefault && hideDefaultsPerClass && targetClass !== undefined && !visibleClasses.has(targetClass)) return false;
        if (!isTargetDefault && targetClass !== undefined && !visibleClasses.has(targetClass)) return false;

        // Edge Weight Checks
        const w = edge.data.weight;
        if (Math.abs(w) < currentThreshold) return false;
        if (w > 0 && !showSupports) return false;
        if (w < 0 && !showAttacks) return false;
        
        return true;
    });

    if (selectedNewCaseIndex !== -1 && newCasesAdjacency.length > 0 && showAttacks) {
        // shape is usually (B, 1, n, d). Wait let's just loop over j
        for (let j = 0; j < numNodes; j++) {
            // Target Node Visibility check
            const isTargetDefault = defaultIndexesData.includes(j);
            const targetScore = baseScoresData[j] ? baseScoresData[j][currentGraphIndex] : 1;
            if (!isTargetDefault && targetScore < currentBaseScoreThreshold) continue;
            
            if (finalStrengths.length > 0) {
                const targetFinalStrength = getFinalStrength(selectedNewCaseIndex, j, currentGraphIndex);
                if (!isTargetDefault && targetFinalStrength < currentFinalStrengthThreshold) continue;
            }
            
            const targetClass = yTrainData[j];
            if (isTargetDefault && !showDefaultCases) continue;
            if (isTargetDefault && hideDefaultsPerClass && targetClass !== undefined && !visibleClasses.has(targetClass)) continue;
            if (!isTargetDefault && targetClass !== undefined && !visibleClasses.has(targetClass)) continue;

            const weight = newCasesAdjacency[selectedNewCaseIndex][j][currentGraphIndex];
            
            if (Math.abs(weight) >= currentThreshold) {
                edges.push({
                    data: {
                        id: `e_nc_${j}_${currentGraphIndex}`,
                        source: 'new_case_node',
                        target: `n${j}`,
                        weight: weight,
                        lineColor: getPurpleColor(weight),
                        width: Math.max(1, Math.abs(weight) * 5)
                    }
                });
            }
        }
    }
    
    return edges;
}

function getNewCaseScore(caseIndex, graphIndex) {
    if (!newCasesBaseScores || newCasesBaseScores[caseIndex] === undefined) return 1;
    const scoreVal = newCasesBaseScores[caseIndex][graphIndex];
    if (Array.isArray(scoreVal)) {
        return scoreVal[0] !== undefined ? scoreVal[0] : 1;
    }
    return scoreVal !== undefined ? scoreVal : 1;
}

function getFinalStrength(caseIndex, nodeId, graphIndex) {
    if (!finalStrengths || finalStrengths[caseIndex] === undefined) return 1;
    
    const nodeData = finalStrengths[caseIndex][nodeId];
    if (nodeData === undefined) return 1;
    
    // Fallback: If it's a number, the graph dimension was squeezed out by the backend
    if (typeof nodeData === 'number') {
        return nodeData;
    }
    
    const strengthVal = nodeData[graphIndex];
    if (Array.isArray(strengthVal)) {
        return strengthVal[0] !== undefined ? strengthVal[0] : 1;
    }
    return strengthVal !== undefined ? strengthVal : 1;
}

/**
 * Returns a Set of node IDs that can reach a default case based on a backwards BFS.
 */
function getNodesConnectedToDefaults(filteredEdges) {
    const visited = new Set();
    const queue = [];

    // Always protect default cases
    defaultIndexesData.forEach(id => {
        visited.add(id);
        queue.push(id);
    });

    // Build reverse adjacency list: target -> source
    const incoming = {};
    filteredEdges.forEach(edge => {
        let s = edge.data.source === 'new_case_node' ? 'new_case_node' : parseInt(edge.data.source.substring(1));
        let t = edge.data.target === 'new_case_node' ? 'new_case_node' : parseInt(edge.data.target.substring(1));
        
        if (!incoming[t]) incoming[t] = [];
        incoming[t].push(s);
    });

    // BFS traversing backwards
    while (queue.length > 0) {
        const curr = queue.shift();
        if (incoming[curr]) {
            incoming[curr].forEach(source => {
                if (!visited.has(source)) {
                    visited.add(source);
                    queue.push(source);
                }
            });
        }
    }
    return visited;
}

/**
 * Initializes Cytoscape (Called ONLY ONCE per file upload)
 */
function initCytoscape() {
    if (cy) {
        cy.destroy();
    }

    const elements = [];
    const uniqueClasses = [...new Set(yTrainData)].sort((a, b) => a - b);

    const currentEdges = getFilteredEdges();
    let connectedNodes = null;
    if (removeSpikes) {
        connectedNodes = getNodesConnectedToDefaults(currentEdges);
    }

    // Add nodes
    for (let i = 0; i < numNodes; i++) {
        const isDefault = defaultIndexesData.includes(i);
        const classInt = yTrainData[i];
        
        let bgColor = '#888';
        if (classInt !== undefined) {
            const classIndex = uniqueClasses.indexOf(classInt);
            bgColor = getClassColor(classIndex, uniqueClasses.length);
        }

        const score = baseScoresData[i] ? baseScoresData[i][currentGraphIndex] : 1;
        let nodeDisplay = 'element';
        let nodeOpacity = 1;

        const fStrength = (selectedNewCaseIndex !== -1 && finalStrengths.length > 0) ? getFinalStrength(selectedNewCaseIndex, i, currentGraphIndex) : score;

        const isVisibleClass = isDefault ? (showDefaultCases && (!hideDefaultsPerClass || classInt === undefined || visibleClasses.has(classInt))) : (classInt !== undefined ? visibleClasses.has(classInt) : true);

        if (!isVisibleClass) {
            nodeDisplay = 'none';
            nodeOpacity = 0;
        } else {
            if ((!isDefault && score < currentBaseScoreThreshold) || 
                (!isDefault && selectedNewCaseIndex !== -1 && finalStrengths.length > 0 && fStrength < currentFinalStrengthThreshold) ||
                (removeSpikes && !isDefault && connectedNodes && !connectedNodes.has(i))) {
                nodeDisplay = 'none';
                nodeOpacity = 0;
            } else {
                if (selectedNewCaseIndex !== -1 && finalStrengths.length > 0 && useTransparentFinalStrength) {
                    let normalizedFStrength = currentMaxFStrength > currentMinFStrength ? (fStrength - currentMinFStrength) / (currentMaxFStrength - currentMinFStrength) : 1;
                    nodeOpacity = 0.10 + (0.90 * normalizedFStrength);
                } else {
                    nodeOpacity = 1;
                }
            }
        }

        let bgImage = 'none';
        let nWidth = 25;
        let nHeight = 25;
        let bWidth = isDefault ? 2 : 0;
        
        if (useBorderBaseScore) {
            bWidth = 1 + (score * 7); // scales from 1 to 8 based on base score
        }
        
        let bColor = '#000';
        let nLabel = isDefault ? `Default ${i}` : `${i}`;
        let nShape = isDefault ? 'square' : 'ellipse';

        if (showImages && hasImages && nodeImages[i]) {
            bgImage = nodeImages[i];
            nWidth = isTabularData ? 480 : 60;
            nHeight = isTabularData ? 120 : 60;
            if (!useBorderBaseScore) {
                bWidth = isDefault ? 6 : 4;
            }
            bColor = bgColor; // Use class color as border
            bgColor = '#fff'; // White fallback behind image
            nShape = isDefault ? 'ellipse' : 'rectangle'; // Use rectangle for images to fit well with borders
        }

        elements.push({ 
            data: { 
                id: `n${i}`, 
                label: nLabel,
                backgroundColor: bgColor,
                shape: nShape,
                borderWidth: bWidth,
                borderColor: bColor,
                opacity: nodeOpacity,
                display: nodeDisplay,
                backgroundImage: bgImage,
                width: nWidth,
                height: nHeight
            } 
        });
    }

    // Add initial edges
    elements.push(...currentEdges);

    // Add New Case Node if selected
    if (selectedNewCaseIndex !== -1) {
        const classInt = parseInt(newCasesLabels[selectedNewCaseIndex]);
        let bgColor = '#888';
        if (!isNaN(classInt)) {
            const classIndex = uniqueClasses.indexOf(classInt);
            bgColor = getClassColor(classIndex, uniqueClasses.length);
        }
        
        const score = getNewCaseScore(selectedNewCaseIndex, currentGraphIndex);
        let nodeOpacity = 0.10 + (0.90 * score);
        // NOTE: new cases don't receive attacks, so their final strength is their base score
        
        let bgImage = 'none';
        let nWidth = 25;
        let nHeight = 25;
        let bWidth = 4;
        if (useBorderBaseScore) {
            bWidth = 1 + (score * 7);
        }
        let bColor = '#8e44ad'; // Purple prominent border
        let nShape = 'diamond';
        let nodeLabel = `New Case ${selectedNewCaseIndex}`;
        if (!isNaN(classInt)) nodeLabel += ` (Class ${classInt})`;

        if (showImages && hasImages && newCaseImages[selectedNewCaseIndex]) {
            bgImage = newCaseImages[selectedNewCaseIndex];
            nWidth = isTabularData ? 560 : 70;
            nHeight = isTabularData ? 140 : 70;
            if (!useBorderBaseScore) {
                bWidth = 6;
            }
            bgColor = '#fff';
            nShape = 'rectangle';
        }

        elements.push({
            data: {
                id: 'new_case_node',
                label: nodeLabel,
                backgroundColor: bgColor,
                shape: nShape,
                borderWidth: bWidth,
                borderColor: bColor,
                opacity: nodeOpacity,
                display: 'element',
                backgroundImage: bgImage,
                width: nWidth,
                height: nHeight
            }
        });
    }

    cy = cytoscape({
        container: document.getElementById('cy'),
        elements: elements,
        style: [
            {
                selector: 'node',
                style: {
                    'background-color': 'data(backgroundColor)',
                    'background-image': 'data(backgroundImage)',
                    'background-fit': 'contain',
                    'background-clip': 'none',
                    'shape': 'data(shape)',
                    'border-width': 'data(borderWidth)',
                    'border-color': 'data(borderColor)',
                    'opacity': 'data(opacity)',
                    'display': 'data(display)',
                    'label': 'data(label)',
                    'color': '#333',
                    'font-size': '10px',
                    'text-valign': showImages ? 'bottom' : 'center',
                    'text-halign': 'center',
                    'width': 'data(width)',
                    'height': 'data(height)'
                }
            },
            {
                selector: 'edge',
                style: {
                    'width': 'data(width)',
                    'line-color': 'data(lineColor)',
                    'target-arrow-color': 'data(lineColor)',
                    'target-arrow-shape': 'triangle',
                    'curve-style': 'bezier',
                    'arrow-scale': 1.5,
                    'opacity': 0.8
                }
            }
        ],
        layout: {
            name: 'dagre',
            rankDir: 'TB',
            nodeSep: currentHorizontalSpacing,
            rankSep: currentVerticalSpacing,
            padding: 20
        }
    });

    cy.on('layoutstop', () => {
        // Layout finished, nodes are free to be dragged
    });

    // Tooltip logic
    cy.on('mouseover', 'node', (e) => {
        const node = e.target;
        
        if (node.id() === 'new_case_node') {
            const classInt = parseInt(newCasesLabels[selectedNewCaseIndex]);
            const score = getNewCaseScore(selectedNewCaseIndex, currentGraphIndex);
            
            let tooltipHtml = `<strong>Type:</strong> New Case<br/>`;
            if (!isNaN(classInt)) tooltipHtml += `<strong>Class:</strong> ${classInt}<br/>`;
            if (score !== null && score !== undefined) tooltipHtml += `<strong>Base Score:</strong> ${score.toFixed(4)}`;
            
            nodeTooltip.innerHTML = tooltipHtml;
            nodeTooltip.classList.remove('hidden');
            return;
        }
        
        const nodeId = parseInt(node.id().substring(1));
        const score = baseScoresData[nodeId] ? baseScoresData[nodeId][currentGraphIndex] : null;
        const isDefault = defaultIndexesData.includes(nodeId);
        const classInt = yTrainData[nodeId];
        
        if (score !== null) {
            let tooltipHtml = isDefault ? `<strong>Type:</strong> Default Case<br/><strong>Class:</strong> ${classInt}<br/>` : `<strong>Class:</strong> ${classInt}<br/>`;
            tooltipHtml += `<strong>Base Score:</strong> ${score.toFixed(4)}`;
            
            if (selectedNewCaseIndex !== -1 && finalStrengths.length > 0) {
                const fStrength = getFinalStrength(selectedNewCaseIndex, nodeId, currentGraphIndex);
                tooltipHtml += `<br/><strong>Final Strength:</strong> ${fStrength.toFixed(4)}`;
            }
            
            nodeTooltip.innerHTML = tooltipHtml;
            nodeTooltip.classList.remove('hidden');
        }
    });

    cy.on('mousemove', 'node', (e) => {
        if (!nodeTooltip.classList.contains('hidden')) {
            // Position tooltip slightly offset from the mouse cursor
            nodeTooltip.style.left = e.originalEvent.clientX + 15 + 'px';
            nodeTooltip.style.top = e.originalEvent.clientY + 15 + 'px';
        }
    });

    cy.on('mouseout', 'node', () => {
        nodeTooltip.classList.add('hidden');
    });

    // Edge click tooltip
    cy.on('click', 'edge', (e) => {
        const edge = e.target;
        const weight = edge.data('weight');

        let tooltipHtml = `<strong>Edge Weight:</strong> ${weight.toFixed(4)}`;
        if (graeAvailableForCurrentCase()) {
            tooltipHtml += `<br/><em>Selected for G-RAE perturbation</em>`;
        }
        nodeTooltip.innerHTML = tooltipHtml;

        nodeTooltip.style.left = e.originalEvent.clientX + 15 + 'px';
        nodeTooltip.style.top = e.originalEvent.clientY + 15 + 'px';
        nodeTooltip.classList.remove('hidden');

        if (graeAvailableForCurrentCase()) {
            const source = edge.data('source');
            const target = edge.data('target');
            const j = parseInt(target.substring(1));

            if (source === 'new_case_node') {
                selectedPerturbationEdge = { type: 'new_case', j };
            } else {
                const i = parseInt(source.substring(1));
                selectedPerturbationEdge = { type: 'casebase', i, j };
            }
            currentGraeDelta = 0;
            graeDeltaSlider.value = 0;
            updateGraePanel();
        }
    });

    // Hide tooltip when clicking the background
    cy.on('click', (e) => {
        if (e.target === cy) {
            nodeTooltip.classList.add('hidden');
        }
    });
}

/**
 * Re-runs the Dagre layout with current spacing settings
 */
function runLayout() {
    if (!cy) return;
    
    cy.layout({
        name: 'dagre',
        rankDir: 'TB',
        nodeSep: currentHorizontalSpacing,
        rankSep: currentVerticalSpacing,
        padding: 20
    }).run();
}

/**
 * Updates both nodes and edges in Cytoscape
 */
function updateGraphElements() {
    if (!cy) return;

    // We need to re-init if the image toggle state changed because text-valign and some core node styles change drastically.
    // However, we want to maintain positions.
    // Instead of completely destroying, we can just update the nodes via cy.batch
    
    const uniqueClasses = [...new Set(yTrainData)].sort((a, b) => a - b);
    const currentEdges = getFilteredEdges();
    let connectedNodes = null;
    if (removeSpikes) {
        connectedNodes = getNodesConnectedToDefaults(currentEdges);
    }

    cy.batch(() => {
        // 1. Update Nodes
        cy.nodes().forEach(node => {
            if (node.id() === 'new_case_node') {
                return; // We will handle new_case_node completely dynamically below
            }

            const nodeId = parseInt(node.id().substring(1));
            const isDefault = defaultIndexesData.includes(nodeId);
            const score = baseScoresData[nodeId] ? baseScoresData[nodeId][currentGraphIndex] : 1;
            const classInt = yTrainData[nodeId];

            const fStrength = (selectedNewCaseIndex !== -1 && finalStrengths.length > 0) ? getFinalStrength(selectedNewCaseIndex, nodeId, currentGraphIndex) : score;

            // Opacity and Display
            const isVisibleClass = isDefault ? (showDefaultCases && (!hideDefaultsPerClass || classInt === undefined || visibleClasses.has(classInt))) : (classInt !== undefined ? visibleClasses.has(classInt) : true);

            if (!isVisibleClass) {
                node.data('display', 'none');
                node.data('opacity', 0);
            } else {
                if ((!isDefault && score < currentBaseScoreThreshold) || 
                    (!isDefault && selectedNewCaseIndex !== -1 && finalStrengths.length > 0 && fStrength < currentFinalStrengthThreshold) ||
                    (removeSpikes && !isDefault && connectedNodes && !connectedNodes.has(nodeId))) {
                    node.data('display', 'none');
                    node.data('opacity', 0);
                } else {
                    node.data('display', 'element');
                    if (selectedNewCaseIndex !== -1 && finalStrengths.length > 0 && useTransparentFinalStrength) {
                        let normalizedFStrength = currentMaxFStrength > currentMinFStrength ? (fStrength - currentMinFStrength) / (currentMaxFStrength - currentMinFStrength) : 1;
                        node.data('opacity', 0.10 + (0.90 * normalizedFStrength));
                    } else {
                        node.data('opacity', 1);
                    }
                }
            }

            // Image Toggle Styles
            let bgColor = '#888';
            if (classInt !== undefined) {
                const classIndex = uniqueClasses.indexOf(classInt);
                bgColor = getClassColor(classIndex, uniqueClasses.length);
            }

            let bgImage = 'none';
            let nWidth = 25;
            let nHeight = 25;
            let bWidth = isDefault ? 2 : 0;
            
            if (useBorderBaseScore) {
                bWidth = 1 + (score * 7);
            }

            let bColor = '#000';
            let nShape = isDefault ? 'square' : 'ellipse';

            if (showImages && hasImages && nodeImages[nodeId]) {
                bgImage = nodeImages[nodeId];
                nWidth = isTabularData ? 480 : 60;
                nHeight = isTabularData ? 120 : 60;
                if (!useBorderBaseScore) {
                    bWidth = isDefault ? 6 : 4;
                }
                bColor = bgColor; 
                bgColor = '#fff'; 
                nShape = isDefault ? 'ellipse' : 'rectangle';
            }

            node.data('backgroundImage', bgImage);
            node.data('width', nWidth);
            node.data('height', nHeight);
            node.data('backgroundColor', bgColor);
            node.data('borderWidth', bWidth);
            node.data('borderColor', bColor);
            node.data('shape', nShape);
            
            // Re-apply style to force text-valign update
            node.style('text-valign', showImages ? 'bottom' : 'center');
        });

        // 2. Remove all current edges
        cy.edges().remove();

        // 3. Handle the new case node dynamically if it needs to be added/removed without a full re-init
        let newCaseNode = cy.getElementById('new_case_node');
        
        // Remove existing new case node to ensure clean state
        if (newCaseNode.length > 0) {
            cy.remove(newCaseNode);
        }

        if (selectedNewCaseIndex !== -1) {
            // Needs to be freshly added
            const classInt = parseInt(newCasesLabels[selectedNewCaseIndex]);
            const score = getNewCaseScore(selectedNewCaseIndex, currentGraphIndex);
            
            let bgColor = '#888';
            if (!isNaN(classInt)) {
                const classIndex = uniqueClasses.indexOf(classInt);
                bgColor = getClassColor(classIndex, uniqueClasses.length);
            }
            
            let bgImage = 'none';
            let nWidth = 25;
            let nHeight = 25;
            let bWidth = 4;
            if (useBorderBaseScore) {
                bWidth = 1 + (score * 7);
            }
            let bColor = '#8e44ad';
            let nShape = 'diamond';
            let nodeLabel = `New Case ${selectedNewCaseIndex}`;
            if (!isNaN(classInt)) nodeLabel += ` (Class ${classInt})`;

            if (showImages && hasImages && newCaseImages[selectedNewCaseIndex]) {
                bgImage = newCaseImages[selectedNewCaseIndex];
                nWidth = isTabularData ? 560 : 70;
                nHeight = isTabularData ? 140 : 70;
                if (!useBorderBaseScore) {
                    bWidth = 6;
                }
                bgColor = '#fff';
                nShape = 'rectangle';
            }

            cy.add({
                data: {
                    id: 'new_case_node',
                    label: nodeLabel,
                    backgroundColor: bgColor,
                    shape: nShape,
                    borderWidth: bWidth,
                    borderColor: bColor,
                    opacity: 0.10 + (0.90 * score),
                    display: 'element',
                    backgroundImage: bgImage,
                    width: nWidth,
                    height: nHeight
                }
            });
            // Force re-style just in case it takes previous cached styles
            cy.getElementById('new_case_node').style('text-valign', showImages ? 'bottom' : 'center');
        }

        // 4. Add the newly filtered edges
        cy.add(currentEdges);
    });
}
