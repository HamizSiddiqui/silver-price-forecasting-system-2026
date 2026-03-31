// Dashboard Dynamic Data & UI Logic

const dashboardData = {
    trainingTime: "March 31, 2026, 03:32 AM",
    currentPrice: 19579.0,
    predictedPrice: 19842.0,
    unit: 'tola'
};

function updateMetrics() {
    document.getElementById('training-time').innerText = dashboardData.trainingTime;
    
    // Base is PKR per Ounce
    const OUNCE_TO_GRAMS = 31.1035;
    const TOLA_TO_GRAMS = 11.6638;
    
    let multiplier = 1;
    let unitLabel = '/ ounce';
    
    if (dashboardData.unit === 'gram') {
        multiplier = 1 / OUNCE_TO_GRAMS;
        unitLabel = '/ gram';
    } else if (dashboardData.unit === 'tola') {
        multiplier = TOLA_TO_GRAMS / OUNCE_TO_GRAMS;
        unitLabel = '/ tola';
    } else if (dashboardData.unit === 'kg') {
        multiplier = 1000 / OUNCE_TO_GRAMS;
        unitLabel = '/ kg';
    }

    const currentElem = document.getElementById('current-price');
    const predictedElem = document.getElementById('predicted-price');

    if (currentElem) currentElem.innerText = (dashboardData.currentPrice * multiplier).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}) + ' ' + unitLabel;
    if (predictedElem) predictedElem.innerText = (dashboardData.predictedPrice * multiplier).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}) + ' ' + unitLabel;
}

function setUnit(unit) {
    dashboardData.unit = unit;
    
    // Update active button
    document.querySelectorAll('.unit-selector .btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.innerText.toLowerCase().includes(unit)) {
            btn.classList.add('active');
        }
    });

    updateMetrics();
    // In a real app, we would also update the graph data here
}

document.addEventListener('DOMContentLoaded', () => {
    updateMetrics();
});
