"""Web-based Review Tool v2 for human verification of seal crops.

Allows users to:
- Load review_queue.csv
- View text crop and full source image side-by-side
- Accept proposed label (Enter)
- Edit label and accept (Type + Enter)
- Reject image (Delete / X / Reject button)
- Resume from existing reviewed_labels.csv
"""

import argparse
import csv
import json
import os
import re
import shutil
import sys
from http import server
from pathlib import Path
from urllib.parse import unquote, urlparse

DEFAULT_PORT = 8520
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Container Seal Review Tool v2</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-main: #0b0f17;
      --bg-panel: #131b26;
      --bg-card: #1c2738;
      --bg-input: #0f1622;
      --border: #2b394f;
      --border-focus: #38bdf8;
      --accent: #2563eb;
      --accent-hover: #1d4ed8;
      --accent-light: #38bdf8;
      --success: #10b981;
      --success-hover: #059669;
      --warning: #f59e0b;
      --danger: #ef4444;
      --danger-hover: #dc2626;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --font-ui: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: var(--font-ui);
      background-color: var(--bg-main);
      color: var(--text-main);
      display: flex;
      height: 100vh;
      overflow: hidden;
      user-select: none;
    }

    .main-section {
      flex: 1;
      display: flex;
      flex-direction: column;
      height: 100vh;
      border-right: 1px solid var(--border);
      position: relative;
    }

    .header-bar {
      height: 56px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 20px;
      background: rgba(19, 27, 38, 0.95);
      border-bottom: 1px solid var(--border);
      backdrop-filter: blur(8px);
      z-index: 10;
    }

    .header-title {
      font-size: 15px;
      font-weight: 700;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .header-badge {
      background: rgba(37, 99, 235, 0.2);
      color: var(--accent-light);
      border: 1px solid rgba(56, 189, 248, 0.3);
      padding: 3px 8px;
      border-radius: 6px;
      font-size: 12px;
      font-family: var(--font-mono);
    }

    .header-actions {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .action-btn {
      background: var(--bg-card);
      border: 1px solid var(--border);
      color: var(--text-main);
      padding: 6px 12px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 13px;
      font-weight: 500;
      transition: all 0.15s ease;
    }

    .action-btn:hover {
      background: #25334a;
      border-color: #3b4e6b;
    }

    .images-grid {
      flex: 1;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      padding: 16px;
      overflow: hidden;
      background: radial-gradient(circle at center, #131c2b 0%, #090d14 100%);
    }

    .image-panel {
      display: flex;
      flex-direction: column;
      background: var(--bg-panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
      position: relative;
    }

    .panel-header {
      padding: 10px 16px;
      background: rgba(28, 39, 56, 0.6);
      border-bottom: 1px solid var(--border);
      font-size: 13px;
      font-weight: 600;
      color: var(--text-muted);
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }

    .image-controls {
      display: flex;
      align-items: center;
      gap: 4px;
      white-space: nowrap;
    }

    .image-control-btn {
      min-width: 24px;
      height: 24px;
      padding: 0 5px;
      border: 1px solid var(--border);
      border-radius: 4px;
      background: var(--bg-card);
      color: var(--text-main);
      cursor: pointer;
      font-size: 12px;
      font-weight: 700;
    }

    .image-control-btn:hover {
      background: #27374f;
      border-color: var(--accent-light);
    }

    .zoom-label {
      min-width: 42px;
      text-align: center;
      color: var(--text-muted);
      font-family: var(--font-mono);
      font-size: 10px;
    }

    .image-viewport {
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      position: relative;
      background: #080c12;
      cursor: grab;
    }

    .image-viewport:active {
      cursor: grabbing;
    }

    .image-viewport img {
      max-width: 90%;
      max-height: 90%;
      object-fit: contain;
      border-radius: 4px;
      box-shadow: 0 4px 20px rgba(0,0,0,0.5);
      transition: transform 0.05s ease-out;
    }

    .sidebar {
      width: 440px;
      height: 100vh;
      display: flex;
      flex-direction: column;
      background: var(--bg-panel);
    }

    .review-form {
      padding: 24px;
      border-bottom: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .counter-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .progress-badge {
      background: var(--accent);
      color: #fff;
      font-size: 12px;
      font-weight: 700;
      padding: 4px 10px;
      border-radius: 6px;
      letter-spacing: 0.5px;
    }

    .status-pill {
      font-size: 12px;
      font-weight: 600;
      padding: 3px 10px;
      border-radius: 12px;
    }

    .status-pill.unreviewed {
      background: rgba(245, 158, 11, 0.15);
      color: var(--warning);
      border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .status-pill.verified {
      background: rgba(16, 185, 129, 0.15);
      color: var(--success);
      border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .status-pill.rejected {
      background: rgba(239, 68, 68, 0.15);
      color: var(--danger);
      border: 1px solid rgba(239, 68, 68, 0.3);
    }

    .meta-box {
      background: var(--bg-input);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
      font-size: 12px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .meta-item {
      display: flex;
      justify-content: space-between;
    }

    .meta-key {
      color: var(--text-muted);
    }

    .meta-val {
      font-family: var(--font-mono);
      color: #cbd5e1;
      font-weight: 600;
      word-break: break-all;
    }

    .reason-box {
      background: rgba(245, 158, 11, 0.1);
      border: 1px solid rgba(245, 158, 11, 0.25);
      border-radius: 8px;
      padding: 10px 14px;
      font-size: 12px;
      color: #fde68a;
      line-height: 1.4;
    }

    .input-label {
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      color: var(--text-muted);
    }

    .label-input {
      width: 100%;
      height: 52px;
      background: var(--bg-input);
      border: 2px solid var(--border);
      border-radius: 8px;
      color: var(--accent-light);
      font-family: var(--font-mono);
      font-size: 22px;
      font-weight: 700;
      text-align: center;
      letter-spacing: 2px;
      outline: none;
      transition: all 0.15s ease;
    }

    .label-input:focus {
      border-color: var(--border-focus);
      box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.2);
    }

    .button-group {
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 10px;
    }

    .btn-verify {
      background: var(--success);
      color: #fff;
      border: none;
      border-radius: 8px;
      padding: 14px;
      font-size: 15px;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      transition: background 0.15s ease;
    }

    .btn-verify:hover {
      background: var(--success-hover);
    }

    .btn-reject {
      background: var(--danger);
      color: #fff;
      border: none;
      border-radius: 8px;
      padding: 14px;
      font-size: 15px;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      transition: background 0.15s ease;
    }

    .btn-reject:hover {
      background: var(--danger-hover);
    }

    .nav-row {
      display: flex;
      justify-content: space-between;
      gap: 10px;
    }

    .nav-btn {
      flex: 1;
      background: var(--bg-card);
      border: 1px solid var(--border);
      color: var(--text-main);
      padding: 10px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 13px;
      font-weight: 600;
      transition: all 0.15s ease;
    }

    .nav-btn:hover {
      background: #27374f;
    }

    .queue-list-section {
      flex: 1;
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }

    .queue-list-header {
      padding: 12px 20px;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--text-muted);
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
    }

    .filter-row {
      display: flex;
      gap: 6px;
      padding: 10px 14px;
      border-bottom: 1px solid var(--border);
      background: rgba(15, 22, 34, 0.65);
    }

    .filter-btn {
      flex: 1;
      border: 1px solid var(--border);
      border-radius: 5px;
      padding: 5px 6px;
      background: var(--bg-card);
      color: var(--text-muted);
      cursor: pointer;
      font-size: 11px;
      font-weight: 600;
    }

    .filter-btn.active {
      color: #fff;
      border-color: var(--accent-light);
      background: rgba(37, 99, 235, 0.45);
    }

    .queue-items {
      flex: 1;
      overflow-y: auto;
      padding: 8px 12px;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }

    .queue-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 8px 12px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 13px;
      border: 1px solid transparent;
      transition: all 0.1s ease;
    }

    .queue-item:hover {
      background: var(--bg-card);
    }

    .queue-item.active {
      background: rgba(37, 99, 235, 0.15);
      border-color: rgba(56, 189, 248, 0.4);
    }

    .item-label {
      font-family: var(--font-mono);
      font-weight: 600;
    }
  </style>
</head>
<body>
  <div class="main-section">
    <div class="header-bar">
      <div class="header-title">
        <span>Container Seal Reviewer</span>
        <span class="header-badge" id="queue-count">0 items</span>
      </div>
      <div class="header-actions">
        <button class="action-btn" id="btn-rot-left">↺ Rotate Crop</button>
        <button class="action-btn" id="btn-reset-zoom">Reset Zoom</button>
      </div>
    </div>

    <div class="images-grid">
      <div class="image-panel">
        <div class="panel-header">
          <span>CHARACTER LINE CROP</span>
          <span id="crop-name" style="font-family: var(--font-mono); font-size: 11px;"></span>
          <div class="image-controls">
            <button class="image-control-btn" data-image="crop" data-action="zoom-out" title="Zoom out">−</button>
            <span class="zoom-label" id="zoom-crop">100%</span>
            <button class="image-control-btn" data-image="crop" data-action="zoom-in" title="Zoom in">+</button>
            <button class="image-control-btn" data-image="crop" data-action="rotate-left" title="Rotate left">↺</button>
            <button class="image-control-btn" data-image="crop" data-action="rotate-right" title="Rotate right">↻</button>
            <button class="image-control-btn" data-image="crop" data-action="reset" title="Reset crop view">Reset</button>
          </div>
        </div>
        <div class="image-viewport" id="viewport-crop">
          <img id="img-crop" src="" alt="Line Crop">
        </div>
      </div>

      <div class="image-panel">
        <div class="panel-header">
          <span>FULL SOURCE IMAGE</span>
          <span id="source-name" style="font-family: var(--font-mono); font-size: 11px;"></span>
          <div class="image-controls">
            <button class="image-control-btn" data-image="source" data-action="zoom-out" title="Zoom out">−</button>
            <span class="zoom-label" id="zoom-source">100%</span>
            <button class="image-control-btn" data-image="source" data-action="zoom-in" title="Zoom in">+</button>
            <button class="image-control-btn" data-image="source" data-action="rotate-left" title="Rotate left">↺</button>
            <button class="image-control-btn" data-image="source" data-action="rotate-right" title="Rotate right">↻</button>
            <button class="image-control-btn" data-image="source" data-action="reset" title="Reset source view">Reset</button>
          </div>
        </div>
        <div class="image-viewport" id="viewport-source">
          <img id="img-source" src="" alt="Source Image">
        </div>
      </div>
    </div>
  </div>

  <div class="sidebar">
    <div class="review-form">
      <div class="counter-row">
        <span class="progress-badge" id="counter-badge">#1 / 0</span>
        <span class="status-pill unreviewed" id="item-status">NEEDS REVIEW</span>
      </div>

      <div class="reason-box" id="reason-text">
        Reason will appear here.
      </div>

      <div class="meta-box">
        <div class="meta-item">
          <span class="meta-key">Source Image:</span>
          <span class="meta-val" id="meta-source">-</span>
        </div>
        <div class="meta-item">
          <span class="meta-key">Crop File:</span>
          <span class="meta-val" id="meta-crop">-</span>
        </div>
      </div>

      <span class="input-label">VERIFIED SEAL NUMBER (A-Z, 0-9)</span>
      <input type="text" class="label-input" id="label-input" autocomplete="off" spellcheck="false" placeholder="SEAL NUMBER">

      <div class="button-group">
        <button class="btn-verify" id="btn-verify">
          <span>✓ Verify (Enter)</span>
        </button>
        <button class="btn-reject" id="btn-reject">
          <span>✕ Reject</span>
        </button>
      </div>

      <div class="nav-row">
        <button class="nav-btn" id="btn-prev">← Prev</button>
        <button class="nav-btn" id="btn-next">Next →</button>
      </div>
    </div>

    <div class="queue-list-section">
      <div class="queue-list-header">
        <span>Review Queue Items</span>
        <span id="reviewed-ratio">0 reviewed</span>
      </div>
      <div class="filter-row">
        <button class="filter-btn" data-filter="ALL">Tất cả</button>
        <button class="filter-btn active" data-filter="REJECTED">✕ Cần review</button>
        <button class="filter-btn" data-filter="VERIFIED">✓ Đã verify</button>
      </div>
      <div class="queue-items" id="queue-items">
      </div>
    </div>
  </div>

  <script>
    let queue = [];
    let reviewed = {};
    let currentIndex = 0;
    let activeQueueEl = null;
    let queueFilter = 'REJECTED';
    const imageState = {
      crop: { scale: 1, rotation: 0 },
      source: { scale: 1, rotation: 0 }
    };

    function itemKey(item) {
      return item.image || `@source:${item.source_image || ''}`;
    }

    const imgCrop = document.getElementById('img-crop');
    const imgSource = document.getElementById('img-source');
    const labelInput = document.getElementById('label-input');
    const reasonText = document.getElementById('reason-text');
    const metaSource = document.getElementById('meta-source');
    const metaCrop = document.getElementById('meta-crop');
    const counterBadge = document.getElementById('counter-badge');
    const itemStatus = document.getElementById('item-status');
    const queueItemsContainer = document.getElementById('queue-items');
    const queueCountBadge = document.getElementById('queue-count');
    const reviewedRatio = document.getElementById('reviewed-ratio');

    async function loadData() {
      try {
        const res = await fetch('/api/queue');
        const data = await res.json();
        queue = data.queue || [];
        reviewed = data.reviewed || {};
        queueCountBadge.innerText = `${queue.length} items`;
        updateFilterCounts();
        renderQueueList();

        // Find first unreviewed
        let firstUnreviewed = queue.findIndex(item => {
          const rev = reviewed[itemKey(item)];
          return !rev || rev.status === 'REJECTED';
        });
        if (firstUnreviewed === -1) firstUnreviewed = 0;
        selectItem(firstUnreviewed);
      } catch (err) {
        alert('Error loading queue: ' + err);
      }
    }

    function selectItem(idx) {
      if (idx < 0 || idx >= queue.length) return;
      currentIndex = idx;
      resetImage('crop');
      resetImage('source');
      updateView();
    }

    function applyImageTransform(kind) {
      const state = imageState[kind];
      const image = kind === 'crop' ? imgCrop : imgSource;
      const zoomLabel = document.getElementById(`zoom-${kind}`);
      image.style.transform = `scale(${state.scale}) rotate(${state.rotation}deg)`;
      zoomLabel.innerText = `${Math.round(state.scale * 100)}%`;
    }

    function resetImage(kind) {
      imageState[kind].scale = 1;
      imageState[kind].rotation = 0;
      applyImageTransform(kind);
    }

    function zoomImage(kind, delta) {
      const state = imageState[kind];
      state.scale = Math.min(4, Math.max(0.25, state.scale + delta));
      applyImageTransform(kind);
    }

    function rotateImage(kind, delta) {
      imageState[kind].rotation = (imageState[kind].rotation + delta + 360) % 360;
      applyImageTransform(kind);
    }

    function updateView() {
      const item = queue[currentIndex];
      if (!item) return;

      counterBadge.innerText = `#${currentIndex + 1} / ${queue.length}`;
      metaSource.innerText = item.source_image;
      metaCrop.innerText = item.image;
      document.getElementById('crop-name').innerText = item.image;
      document.getElementById('source-name').innerText = item.source_image;

      imgCrop.src = `/api/image/crop?path=${encodeURIComponent(item.image)}`;
      imgSource.src = `/api/image/source?name=${encodeURIComponent(item.source_image)}`;
      applyImageTransform('crop');
      applyImageTransform('source');

      reasonText.innerText = item.reason || 'Quality gate review requested.';

      const rev = reviewed[itemKey(item)];
      if (rev) {
        labelInput.value = rev.proposed_label;
        if (rev.status === 'VERIFIED') {
          itemStatus.className = 'status-pill verified';
          itemStatus.innerText = 'VERIFIED';
        } else {
          itemStatus.className = 'status-pill rejected';
          itemStatus.innerText = 'REJECTED';
        }
      } else {
        labelInput.value = item.proposed_label || '';
        itemStatus.className = 'status-pill unreviewed';
        itemStatus.innerText = 'NEEDS REVIEW';
      }

      labelInput.focus();
      labelInput.select();

      highlightActiveQueueItem();
      updateProgressRatio();
    }

    function renderQueueList() {
      queueItemsContainer.innerHTML = '';
      queue.forEach((item, idx) => {
        const rev = reviewed[itemKey(item)];
        if (!matchesFilter(item, queueFilter)) return;
        const row = document.createElement('div');
        row.className = 'queue-item' + (idx === currentIndex ? ' active' : '');
        row.id = `q-item-${idx}`;
        let statusBadge = '⏳';
        if (rev) {
          statusBadge = rev.status === 'VERIFIED' ? '✓' : '✕';
        }
        row.innerHTML = `
          <div style="display: flex; gap: 8px; align-items: center;">
            <span style="color: var(--text-muted); font-size: 11px;">#${idx + 1}</span>
            <span class="item-label">${item.proposed_label || '(empty)'}</span>
          </div>
          <span data-status style="font-size: 11px;">${statusBadge}</span>
        `;
        row.addEventListener('click', () => selectItem(idx));
        queueItemsContainer.appendChild(row);
      });
    }

    function matchesFilter(item, filter) {
      if (filter === 'ALL') return true;
      const rev = reviewed[itemKey(item)];
      if (filter === 'REJECTED') return !rev || rev.status === 'REJECTED';
      return !!rev && rev.status === filter;
    }

    function filteredCount(filter) {
      return queue.reduce((count, item) => count + (matchesFilter(item, filter) ? 1 : 0), 0);
    }

    function updateFilterCounts() {
      const labels = {
        ALL: `Tất cả (${filteredCount('ALL')})`,
        REJECTED: `✕ Cần review (${filteredCount('REJECTED')})`,
        VERIFIED: `✓ Đã verify (${filteredCount('VERIFIED')})`
      };
      document.querySelectorAll('[data-filter]').forEach(btn => {
        btn.innerText = labels[btn.dataset.filter];
      });
    }

    function findFilteredIndex(start, direction) {
      for (let idx = start + direction; idx >= 0 && idx < queue.length; idx += direction) {
        if (matchesFilter(queue[idx], queueFilter)) return idx;
      }
      return -1;
    }

    function moveFiltered(direction) {
      const next = findFilteredIndex(currentIndex, direction);
      if (next >= 0) selectItem(next);
    }

    function setQueueFilter(filter) {
      queueFilter = filter;
      document.querySelectorAll('[data-filter]').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.filter === filter);
      });
      updateFilterCounts();
      renderQueueList();
      const first = queue.findIndex(item => matchesFilter(item, filter));
      if (first >= 0) selectItem(first);
    }

    function setQueueStatus(idx, status) {
      const row = document.getElementById(`q-item-${idx}`);
      const badge = row?.querySelector('[data-status]');
      if (badge) badge.innerText = status === 'VERIFIED' ? '✓' : '✕';
    }

    function highlightActiveQueueItem() {
      activeQueueEl?.classList.remove('active');
      activeQueueEl = document.getElementById(`q-item-${currentIndex}`);
      activeQueueEl?.classList.add('active');
      activeQueueEl?.scrollIntoView({ block: 'nearest' });
    }

    function updateProgressRatio() {
      const reviewedCount = Object.keys(reviewed).length;
      reviewedRatio.innerText = `${reviewedCount} / ${queue.length} reviewed`;
    }

    async function submitReview(status) {
      const item = queue[currentIndex];
      if (!item) return;

      const label = labelInput.value.trim().toUpperCase().replace(/[^A-Z0-9]/g, '');
      if (status === 'VERIFIED' && !label) {
        alert('Label cannot be empty for VERIFIED status!');
        return;
      }

      try {
        const res = await fetch('/api/submit', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            image: item.image,
            proposed_label: label,
            status: status,
            reason: item.reason,
            source_image: item.source_image
          })
        });
        const result = await res.json();
        if (result.ok) {
          reviewed[itemKey(item)] = {
            proposed_label: label,
            status: status
          };
          setQueueStatus(currentIndex, status);
          updateProgressRatio();
          updateFilterCounts();
          if (queueFilter !== 'ALL' && status !== queueFilter) {
            renderQueueList();
            const nextFiltered = findFilteredIndex(currentIndex, 1);
            if (nextFiltered >= 0) selectItem(nextFiltered);
            else updateView();
            return;
          }
          if (queueFilter === 'ALL') moveFiltered(1);
          else moveFiltered(1);
        } else {
          alert('Save error: ' + result.error);
        }
      } catch (err) {
        alert('Network error: ' + err);
      }
    }

    document.getElementById('btn-verify').addEventListener('click', () => submitReview('VERIFIED'));
    document.getElementById('btn-reject').addEventListener('click', () => submitReview('REJECTED'));
    document.querySelectorAll('[data-filter]').forEach(btn => {
      btn.addEventListener('click', () => setQueueFilter(btn.dataset.filter));
    });

    document.getElementById('btn-prev').addEventListener('click', () => {
      moveFiltered(-1);
    });

    document.getElementById('btn-next').addEventListener('click', () => {
      moveFiltered(1);
    });

    document.getElementById('btn-rot-left').addEventListener('click', () => rotateImage('crop', -90));
    document.getElementById('btn-reset-zoom').addEventListener('click', () => resetImage('crop'));
    document.querySelectorAll('[data-image][data-action]').forEach(btn => {
      btn.addEventListener('click', () => {
        const kind = btn.dataset.image;
        const action = btn.dataset.action;
        if (action === 'zoom-in') zoomImage(kind, 0.25);
        else if (action === 'zoom-out') zoomImage(kind, -0.25);
        else if (action === 'rotate-left') rotateImage(kind, -90);
        else if (action === 'rotate-right') rotateImage(kind, 90);
        else if (action === 'reset') resetImage(kind);
      });
    });
    ['crop', 'source'].forEach(kind => {
      document.getElementById(`viewport-${kind}`).addEventListener('wheel', (event) => {
        event.preventDefault();
        zoomImage(kind, event.deltaY < 0 ? 0.25 : -0.25);
      }, { passive: false });
    });

    window.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        submitReview('VERIFIED');
      } else if (e.key === 'Delete' && (e.ctrlKey || e.altKey)) {
        e.preventDefault();
        submitReview('REJECTED');
      } else if (e.key === 'ArrowLeft' && (e.ctrlKey || e.altKey)) {
        e.preventDefault();
        if (currentIndex > 0) selectItem(currentIndex - 1);
      } else if (e.key === 'ArrowRight' && (e.ctrlKey || e.altKey)) {
        e.preventDefault();
        if (currentIndex < queue.length - 1) selectItem(currentIndex + 1);
      }
    });

    loadData();
  </script>
</body>
</html>
"""


def resolve_source_path(source_dir: Path, relative_name: str) -> Path | None:
    """Resolve a v3 relative source path without allowing traversal."""
    root = source_dir.resolve()
    target = (root / relative_name.replace("/", os.sep)).resolve()
    if target.is_relative_to(root) and target.is_file():
        return target
    if "/" not in relative_name and "\\" not in relative_name:
        for split in ("train", "val", "test"):
            legacy = (root / split / "images" / relative_name).resolve()
            if legacy.is_relative_to(root) and legacy.is_file():
                return legacy
    return None


class ReviewServer(server.SimpleHTTPRequestHandler):
    work_dir: Path
    source_dir: Path

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        super().end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in {"/", "/index.html"}:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
            return

        elif path == "/api/queue":
            queue_file = self.work_dir / "review_queue.csv"
            reviewed_file = self.work_dir / "reviewed_labels.csv"

            queue = []
            if queue_file.is_file():
                with queue_file.open(newline="", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    queue = list(reader)

            reviewed = {}
            if reviewed_file.is_file():
                with reviewed_file.open(newline="", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        key = row.get("image", "") or f"@source:{row.get('source_image', '')}"
                        reviewed[key] = {
                            "proposed_label": row.get("proposed_label", ""),
                            "status": row.get("status", "VERIFIED")
                        }

            data = json.dumps({"queue": queue, "reviewed": reviewed}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        elif path == "/api/image/crop":
            query = unquote(parsed.query)
            params = dict(q.split("=", 1) for q in query.split("&") if "=" in q)
            crop_path = params.get("path", "")
            target = (self.work_dir / crop_path).resolve()
            if not target.is_relative_to(self.work_dir):
                target = Path()
            if not target.is_file():
                # Fallback to images dir
                target = (self.work_dir / "images" / Path(crop_path).name).resolve()
            if target.is_relative_to(self.work_dir) and target.is_file():
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(target.stat().st_size))
                self.end_headers()
                self.wfile.write(target.read_bytes())
                return
            self.send_error(404, f"Crop image not found: {crop_path}")
            return

        elif path == "/api/image/source":
            query = unquote(parsed.query)
            params = dict(q.split("=", 1) for q in query.split("&") if "=" in q)
            name = params.get("name", "")
            target = resolve_source_path(self.source_dir, name)
            if target and target.is_file():
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(target.stat().st_size))
                self.end_headers()
                self.wfile.write(target.read_bytes())
                return
            self.send_error(404, f"Source image not found: {name}")
            return

        super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/submit-batch":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8"))
            reviews = payload.get("reviews", [])
            reviewed_file = self.work_dir / "reviewed_labels.csv"
            existing = {}
            if reviewed_file.is_file():
                with reviewed_file.open(newline="", encoding="utf-8-sig") as f:
                    for row in csv.DictReader(f):
                        key = row.get("image", "") or f"@source:{row.get('source_image', '')}"
                        existing[key] = row
            for item in reviews:
                image = str(item.get("image", ""))
                source_image = str(item.get("source_image", ""))
                status = str(item.get("status", "VERIFIED"))
                label = str(item.get("proposed_label", item.get("label", ""))).strip().upper()
                if (not image and not source_image) or status not in {"VERIFIED", "REJECTED"}:
                    continue
                key = image or f"@source:{source_image}"
                existing[key] = {
                    "image": image,
                    "proposed_label": label,
                    "status": status,
                    "reason": item.get("reason", ""),
                    "source_image": source_image
                }
            with reviewed_file.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["image", "proposed_label", "status", "reason", "source_image"])
                writer.writeheader()
                writer.writerows(existing.values())
            resp = json.dumps({"ok": True, "count": len(reviews)}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
            return

        if parsed.path == "/api/submit":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8"))

            reviewed_file = self.work_dir / "reviewed_labels.csv"
            existing = {}
            if reviewed_file.is_file():
                with reviewed_file.open(newline="", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        key = row.get("image", "") or f"@source:{row.get('source_image', '')}"
                        existing[key] = row

            image = payload.get("image", "")
            source_image = payload.get("source_image", "")
            key = image or f"@source:{source_image}"
            existing[key] = {
                "image": payload["image"],
                "proposed_label": payload["proposed_label"],
                "status": payload["status"],
                "reason": payload.get("reason", ""),
                "source_image": source_image
            }

            with reviewed_file.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["image", "proposed_label", "status", "reason", "source_image"])
                writer.writeheader()
                writer.writerows(existing.values())

            resp = json.dumps({"ok": True}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
            return

        self.send_error(404)


def run_reviewer(work_dir: Path, source_dir: Path, port: int = DEFAULT_PORT):
    ReviewServer.work_dir = work_dir.resolve()
    ReviewServer.source_dir = source_dir.resolve()
    with server.HTTPServer(("0.0.0.0", port), ReviewServer) as httpd:
        print(f"\n=======================================================")
        print(f" Review Tool v2 running at: http://localhost:{port}")
        print(f" Work Dir: {work_dir}")
        print(f" Source Dir: {source_dir}")
        print(f"=======================================================\n")
        try:
            import webbrowser
            webbrowser.open(f"http://localhost:{port}")
        except Exception:
            pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down reviewer.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=Path("recognition_dataset_v2_work"))
    parser.add_argument("--source-dir", type=Path, default=Path(os.getenv("SOURCE_DIR", "data/new_dataset")))
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    run_reviewer(args.work_dir, args.source_dir, args.port)
