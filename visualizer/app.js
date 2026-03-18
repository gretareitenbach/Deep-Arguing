// Initialize the cytoscape dagre extension
cytoscape.use(cytoscapeDagre);

// State variables
let numNodes = 0;           // n
let numGraphs = 0;          // d
let currentGraphIndex = 0;
let currentThreshold = 0.3; // Edge Threshold
let currentBaseScoreThreshold = 0; // Node Base Score Threshold
let cy = null;
let baseNodePositions = null; // Cache original layout coordinates
let currentSpacing = 0.8;     // Initial node spacing scale
let showSupports = true;
let showAttacks = true;
let useTransparentNodes = true; // Toggle for node opacity based on score
let showImages = false;     // Toggle to render images
let hasImages = false;      // True if X_train is 3D/4D

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

// UI Elements
const jsonUpload = document.getElementById('json-upload');
const graphSlider = document.getElementById('graph-slider');
const thresholdSlider = document.getElementById('threshold-slider');
const baseScoreSlider = document.getElementById('base-score-slider');
const supportsCheckbox = document.getElementById('supports-checkbox');
const attacksCheckbox = document.getElementById('attacks-checkbox');
const transparentNodesCheckbox = document.getElementById('transparent-nodes-checkbox');
const showImagesCheckbox = document.getElementById('show-images-checkbox');
const imageToggleContainer = document.getElementById('image-toggle-container');

const graphLabel = document.getElementById('graph-label');
const thresholdLabel = document.getElementById('threshold-label');
const baseScoreLabel = document.getElementById('base-score-label');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingText = document.getElementById('loading-text');
const legendContainer = document.getElementById('legend-container');
const legendItems = document.getElementById('legend-items');
const nodeTooltip = document.getElementById('node-tooltip');

let debounceTimeout = null;

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
    legendContainer.style.display = 'flex'; // Show container

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
 * Detects if X_train contains images and precomputes Base64 URIs.
 * Expects shape: (n, H, W) or (n, C, H, W) or (n, H, W, C)
 */
function precomputeImages() {
    hasImages = false;
    nodeImages = {};
    if (!xTrainData || xTrainData.length === 0) return;

    const sample = xTrainData[0];
    
    // Check array depth to infer shape
    let isImage = false;
    let isGrayscale = true;
    let H = 0, W = 0;

    if (Array.isArray(sample) && Array.isArray(sample[0])) {
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

    if (!isImage) return;

    hasImages = true;
    imageToggleContainer.style.display = 'flex';
    showImagesCheckbox.disabled = false;
    showImagesCheckbox.checked = false;

    // Create an off-screen canvas for rendering
    // Scale up small images (like 32x32 CIFAR) directly on the canvas using nearest-neighbor
    // to avoid browser anti-aliasing blur when Cytoscape draws them.
    const SCALE = 2; 
    const canvas = document.createElement('canvas');
    canvas.width = W * SCALE;
    canvas.height = H * SCALE;
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(canvas.width, canvas.height);

    for (let n = 0; n < xTrainData.length; n++) {
        const item = xTrainData[n];
        
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
        nodeImages[n] = canvas.toDataURL('image/png');
    }
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

                    // Initialize filter states
                    visibleClasses = new Set(yTrainData);
                    showDefaultCases = true;

                    setTimeout(() => {
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
                        
                        // Reset states
                        currentGraphIndex = 0;
                        currentThreshold = 0.3;
                        currentBaseScoreThreshold = 0.0;
                        currentSpacing = 0.8;
                        showSupports = true;
                        showAttacks = true;
                        useTransparentNodes = true;
                        showImages = false;
                        baseNodePositions = null;
                        
                        // Update UI Controls
                        graphSlider.min = 0;
                        graphSlider.max = numGraphs - 1;
                        graphSlider.value = 0;
                        graphSlider.disabled = false;
                        
                        thresholdSlider.value = 0.3;
                        thresholdSlider.disabled = false;

                        baseScoreSlider.value = 0.0;
                        baseScoreSlider.disabled = false;

                        supportsCheckbox.checked = true;
                        supportsCheckbox.disabled = false;

                        attacksCheckbox.checked = true;
                        attacksCheckbox.disabled = false;
                        
                        transparentNodesCheckbox.checked = true;
                        transparentNodesCheckbox.disabled = false;
                        
                        updateLabels();
                        showLoading("Rendering graph...");
                        
                        setTimeout(() => {
                            initCytoscape();
                            hideLoading();
                        }, 50);
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
    updateLabels();
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

supportsCheckbox.addEventListener('change', (e) => {
    showSupports = e.target.checked;
    if (precomputedEdges.length > 0) handleSliderChange();
});

attacksCheckbox.addEventListener('change', (e) => {
    showAttacks = e.target.checked;
    if (precomputedEdges.length > 0) handleSliderChange();
});

transparentNodesCheckbox.addEventListener('change', (e) => {
    useTransparentNodes = e.target.checked;
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
}

/**
 * Filters the precomputed edges based on threshold and edge type
 */
function getFilteredEdges() {
    return precomputedEdges[currentGraphIndex].filter(edge => {
        // Source/Target Base Score Visibility check
        const sourceId = parseInt(edge.data.source.substring(1));
        const targetId = parseInt(edge.data.target.substring(1));
        
        const isSourceDefault = defaultIndexesData.includes(sourceId);
        const isTargetDefault = defaultIndexesData.includes(targetId);
        
        const sourceScore = baseScoresData[sourceId] ? baseScoresData[sourceId][currentGraphIndex] : 1;
        const targetScore = baseScoresData[targetId] ? baseScoresData[targetId][currentGraphIndex] : 1;

        if (!isSourceDefault && sourceScore < currentBaseScoreThreshold) return false;
        if (!isTargetDefault && targetScore < currentBaseScoreThreshold) return false;

        // Class Filter Check
        const sourceClass = yTrainData[sourceId];
        const targetClass = yTrainData[targetId];
        
        if (isSourceDefault && !showDefaultCases) return false;
        if (!isSourceDefault && sourceClass !== undefined && !visibleClasses.has(sourceClass)) return false;
        
        if (isTargetDefault && !showDefaultCases) return false;
        if (!isTargetDefault && targetClass !== undefined && !visibleClasses.has(targetClass)) return false;

        // Edge Weight Checks
        const w = edge.data.weight;
        if (Math.abs(w) < currentThreshold) return false;
        if (w > 0 && !showSupports) return false;
        if (w < 0 && !showAttacks) return false;
        
        return true;
    });
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
        let nodeOpacity = score;

        const isVisibleClass = isDefault ? showDefaultCases : (classInt !== undefined ? visibleClasses.has(classInt) : true);

        if (!isVisibleClass) {
            nodeDisplay = 'none';
            nodeOpacity = 0;
        } else if (!isDefault) {
            if (score < currentBaseScoreThreshold) {
                nodeDisplay = 'none';
                nodeOpacity = 0;
            } else {
                nodeOpacity = useTransparentNodes ? score : 1;
            }
        } else {
            nodeOpacity = useTransparentNodes ? Math.max(0.1, score) : 1; 
        }

        let bgImage = 'none';
        let nWidth = 25;
        let nHeight = 25;
        let bWidth = isDefault ? 2 : 0;
        let bColor = '#000';
        let nLabel = isDefault ? `Default ${i}` : `${i}`;
        let nShape = isDefault ? 'square' : 'ellipse';

        if (showImages && hasImages && nodeImages[i]) {
            bgImage = nodeImages[i];
            nWidth = 60;
            nHeight = 60;
            bWidth = isDefault ? 6 : 4;
            bColor = bgColor; // Use class color as border
            bgColor = '#fff'; // White fallback behind image
            nShape = 'rectangle'; // Use rectangle for images to fit well with borders
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
    elements.push(...getFilteredEdges());

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
            nodeSep: 30, // tighter default horizontal spacing
            rankSep: 40, // tighter default vertical spacing
            padding: 20
        }
    });

    // Save initial positions and apply default scale
    cy.on('layoutstop', () => {
        if (!baseNodePositions) {
            baseNodePositions = {};
            cy.nodes().forEach(node => {
                baseNodePositions[node.id()] = { ...node.position() };
            });
            
            // Prevent user from dragging nodes around, but allow API movement
            cy.nodes().ungrabify();
            
            updateNodePositions(); // Apply the default 0.8x spacing immediately
        }
    });

    // Tooltip logic
    cy.on('mouseover', 'node', (e) => {
        const node = e.target;
        const nodeId = parseInt(node.id().substring(1));
        const score = baseScoresData[nodeId] ? baseScoresData[nodeId][currentGraphIndex] : null;
        const isDefault = defaultIndexesData.includes(nodeId);
        const classInt = yTrainData[nodeId];
        
        if (score !== null) {
            let tooltipHtml = isDefault ? `<strong>Type:</strong> Default Case<br/>` : `<strong>Class:</strong> ${classInt}<br/>`;
            tooltipHtml += `<strong>Base Score:</strong> ${score.toFixed(4)}`;
            
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
}

/**
 * Applies the spacing scale to the frozen node layout
 */
function updateNodePositions() {
    if (!cy || !baseNodePositions) return;

    // Calculate center of graph to scale from
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const pos of Object.values(baseNodePositions)) {
        if (pos.x < minX) minX = pos.x;
        if (pos.x > maxX) maxX = pos.x;
        if (pos.y < minY) minY = pos.y;
        if (pos.y > maxY) maxY = pos.y;
    }
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;

    // Apply scale relative to center
    cy.batch(() => {
        cy.nodes().forEach(node => {
            const base = baseNodePositions[node.id()];
            if (base) {
                node.position({
                    x: centerX + (base.x - centerX) * currentSpacing,
                    y: centerY + (base.y - centerY) * currentSpacing
                });
            }
        });
    });

    cy.fit(cy.nodes(), 30);
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

    cy.batch(() => {
        // 1. Update Nodes
        cy.nodes().forEach(node => {
            const nodeId = parseInt(node.id().substring(1));
            const isDefault = defaultIndexesData.includes(nodeId);
            const score = baseScoresData[nodeId] ? baseScoresData[nodeId][currentGraphIndex] : 1;
            const classInt = yTrainData[nodeId];

            // Opacity and Display
            const isVisibleClass = isDefault ? showDefaultCases : (classInt !== undefined ? visibleClasses.has(classInt) : true);

            if (!isVisibleClass) {
                node.data('display', 'none');
                node.data('opacity', 0);
            } else if (!isDefault) {
                if (score < currentBaseScoreThreshold) {
                    node.data('display', 'none');
                    node.data('opacity', 0);
                } else {
                    node.data('display', 'element');
                    node.data('opacity', useTransparentNodes ? score : 1);
                }
            } else {
                node.data('display', 'element');
                node.data('opacity', useTransparentNodes ? Math.max(0.1, score) : 1);
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
            let bColor = '#000';
            let nShape = isDefault ? 'square' : 'ellipse';

            if (showImages && hasImages && nodeImages[nodeId]) {
                bgImage = nodeImages[nodeId];
                nWidth = 60;
                nHeight = 60;
                bWidth = isDefault ? 6 : 4;
                bColor = bgColor; 
                bgColor = '#fff'; 
                nShape = 'rectangle';
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

        // 3. Add the newly filtered edges
        cy.add(getFilteredEdges());
    });
}
