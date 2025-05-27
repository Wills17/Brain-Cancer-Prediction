document.addEventListener('DOMContentLoaded', function () {
  // ======================= THEME TOGGLING =======================
  const themeToggle = document.getElementById('theme-toggle');
  const savedTheme = localStorage.getItem('theme'); // ← Load saved theme from localStorage
  if (savedTheme) {
    document.documentElement.classList.add(savedTheme);
  } else {
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    document.documentElement.classList.add(prefersDark ? 'dark-mode' : 'light-mode'); // ← Set default theme
  }

  if (themeToggle) {
    themeToggle.addEventListener('click', function () {
      const isDark = document.documentElement.classList.toggle('dark-mode');
      document.documentElement.classList.toggle('light-mode', !isDark);
      localStorage.setItem('theme', isDark ? 'dark-mode' : 'light-mode'); // ← Save theme toggle state
    });
  }

  // ======================= TAB SWITCHING =======================
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');
  const uploadTab = document.getElementById('upload-tab'); // ← Cache DOM elements
  const cameraTab = document.getElementById('camera-tab');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', function () {
      const tab = this.getAttribute('data-tab');
      tabBtns.forEach(btn => btn.classList.remove('active'));
      tabContents.forEach(content => content.classList.remove('active'));

      this.classList.add('active');
      if (tab === 'upload') {
        uploadTab.classList.add('active');
      } else {
        cameraTab.classList.add('active');
      }
    });
  });

  if (window.location.pathname.includes('predict')) initPredictPage();
  if (window.location.pathname.includes('results')) loadResults();
});

function initPredictPage() {
  const fileInput = document.getElementById('file-input');
  const uploadContainer = document.getElementById('upload-container');
  const fileInfo = document.getElementById('file-info');
  const fileName = document.getElementById('file-name');
  const fileSize = document.getElementById('file-size');
  const filePreview = document.getElementById('file-preview-image');
  const largePreviewImg = document.getElementById('large-preview-image');
  const removeFileBtn = document.getElementById('remove-file');
  const analyzeBtn = document.getElementById('analyze-btn');
  const previewSection = document.getElementById('preview-section');
  const confirmationPrompt = document.getElementById('confirmation-prompt');
  const confirmYes = document.getElementById('confirm-yes');
  const confirmNo = document.getElementById('confirm-no');

  if (fileInput) {
    fileInput.addEventListener('change', function () {
      const file = this.files[0];
      if (!file || !file.type.startsWith('image/')) { // ← Validate file type
        alert('Please upload a valid image file.');
        return;
      }

      fileName.textContent = file.name;
      fileSize.textContent = Math.round(file.size / 1024) + ' KB';

      const reader = new FileReader();
      reader.onload = function (e) {
        filePreview.src = e.target.result;
        largePreviewImg.src = e.target.result;
      };
      reader.readAsDataURL(file);

      uploadContainer.style.display = 'none';
      fileInfo.style.display = 'flex';
      confirmationPrompt.style.display = 'block';
    });
  }

  if (confirmYes) {
    confirmYes.addEventListener('click', function () {
      confirmationPrompt.style.display = 'none';
      previewSection.style.display = 'block';
      analyzeBtn.disabled = false;
    });
  }

  if (confirmNo) {
    confirmNo.addEventListener('click', function () {
      if (removeFileBtn) {
        removeFileBtn.click();
      }
      confirmationPrompt.style.display = 'none';
    });
  }

  if (removeFileBtn) {
    removeFileBtn.addEventListener('click', function () {
      fileInput.value = '';
      uploadContainer.style.display = 'flex';
      fileInfo.style.display = 'none';
      previewSection.style.display = 'none';
      confirmationPrompt.style.display = 'none';
      analyzeBtn.disabled = true;
      analyzeBtn.textContent = 'Analyze'; // ← Reset text
    });
  }

  if (analyzeBtn) {
    analyzeBtn.addEventListener('click', function () {
      if (analyzeBtn.disabled) return;
      analyzeBtn.textContent = 'Analyzing...';
      analyzeBtn.disabled = true;

      // Prepare the image file for upload
      const file = fileInput.files[0];
      if (!file) {
        alert('No file selected.');
        return;
      }

      const formData = new FormData();
      formData.append('file', file);
      
      fetch('/predict', {
        method: 'POST',
        body: formData
      })
        .then(response => response.json())
        .then(result => {
          localStorage.setItem('brainScanResult', JSON.stringify({
            prediction: result.prediction,
            predictionClass: result.predictionClass,
            imageUrl: largePreviewImg.src,
            description: result.description
          }));
          window.location.href = "/results";
        })
        .catch(error => {
          alert('Prediction failed: ' + error);
          analyzeBtn.textContent = 'Analyze';
          analyzeBtn.disabled = false;
        });
    });
  }

  // ============== Drag & Drop Upload ==============
  const dropArea = document.getElementById('upload-container');
  if (dropArea) {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
      dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
      e.preventDefault();
      e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
      dropArea.addEventListener(eventName, () => dropArea.classList.add('highlight'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropArea.addEventListener(eventName, () => dropArea.classList.remove('highlight'), false);
    });

    dropArea.addEventListener('drop', function (e) {
      const dt = e.dataTransfer;
      const files = dt.files;
      if (files && files.length) {
        fileInput.files = files;
        fileInput.dispatchEvent(new Event('change', { bubbles: true }));
      }
    });
  }
}

// ======================= RESULTS PAGE =======================
function loadResults() {
  const resultImage = document.getElementById('result-image');
  const resultPrediction = document.getElementById('result-prediction');
  const resultDescription = document.getElementById('result-description');
  const resultIndicator = document.getElementById('result-indicator');
  const noResults = document.getElementById('no-results');
  const resultsCard = document.getElementById('results-card');
  const startOverBtn = document.getElementById('start-over-btn');
  const downloadReportBtn = document.getElementById('download-report-btn');

  const storedResult = localStorage.getItem('brainScanResult');

  if (storedResult) {
    const result = JSON.parse(storedResult);

    if (resultImage) resultImage.src = result.imageUrl;
    if (resultPrediction) {
      resultPrediction.textContent = result.prediction;
      resultPrediction.className = 'result-value ' + result.predictionClass;
    }
    if (resultDescription) resultDescription.textContent = result.description;
    if (resultIndicator) resultIndicator.className = 'result-indicator ' + result.predictionClass;

    if (noResults) noResults.style.display = 'none';
    if (resultsCard) resultsCard.style.display = 'grid';
  } else {
    if (noResults) noResults.style.display = 'block';
    if (resultsCard) resultsCard.style.display = 'none';
  }

  if (startOverBtn) {
    startOverBtn.addEventListener('click', function () {
      localStorage.removeItem('brainScanResult');
    });
  }

  if (downloadReportBtn) {
    downloadReportBtn.addEventListener('click', function () {
      const stored = localStorage.getItem('brainScanResult');
      if (!stored) return;

      const result = JSON.parse(stored);
      const report = `
        Brain Scan Report
        -----------------
        Prediction: ${result.prediction}
        Description: ${result.description}
      `;

      const blob = new Blob([report], { type: 'text/plain' });
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = 'brain_scan_report.txt';
      link.click();
    }); // ← Implemented report download using Blob
  }
}
