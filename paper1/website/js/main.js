/* =============================================================================
   C-SE for SC-BSS · Paper 1 — Charts and interactions
   Chart.js 4.x | KaTeX auto-render
   ============================================================================= */
(function () {
  'use strict';

  // ---------------------------------------------------------------------------
  // Theme
  // ---------------------------------------------------------------------------
  const FONT_SERIF = "'Source Serif 4', 'Source Serif Pro', Georgia, serif";
  const FONT_SANS  = "'Inter', -apple-system, BlinkMacSystemFont, sans-serif";
  const FONT_MONO  = "'JetBrains Mono', monospace";

  const COLOR = {
    navy:   '#0F2A47',
    gold:   '#C8964A',
    gold2:  '#A87A33',
    blue:   '#5B8DBE',
    blue2:  '#3E6B89',
    purple: '#7B6FA0',
    terra:  '#C66B5A',
    gray:   '#888888',
    gray2:  '#A89C8B',
    teal:   '#3E6B89',
    ink:    '#1F1B16',
    ink2:   '#4A423A',
    ink3:   '#7A6F62',
    line:   '#D8CFBC',
  };

  // Common Chart.js defaults
  Chart.defaults.font.family = FONT_SANS;
  Chart.defaults.font.size = 12.5;
  Chart.defaults.color = COLOR.ink2;
  Chart.defaults.borderColor = COLOR.line;
  Chart.defaults.plugins.legend.position = 'bottom';
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
  Chart.defaults.plugins.legend.labels.boxWidth = 8;
  Chart.defaults.plugins.legend.labels.boxHeight = 8;
  Chart.defaults.plugins.legend.labels.padding = 14;
  Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(31, 27, 22, 0.96)';
  Chart.defaults.plugins.tooltip.titleFont = { family: FONT_SANS, size: 12, weight: '600' };
  Chart.defaults.plugins.tooltip.bodyFont = { family: FONT_SANS, size: 12 };
  Chart.defaults.plugins.tooltip.padding = 10;
  Chart.defaults.plugins.tooltip.cornerRadius = 6;
  Chart.defaults.plugins.tooltip.boxPadding = 6;
  Chart.defaults.elements.line.borderWidth = 2.2;
  Chart.defaults.elements.point.radius = 4;
  Chart.defaults.elements.point.hoverRadius = 6;
  Chart.defaults.elements.point.borderWidth = 0;
  Chart.defaults.maintainAspectRatio = false;
  Chart.defaults.responsive = true;

  // Axis defaults helper
  function makeScales(opts = {}) {
    const xGrid = { display: opts.xGrid !== false, color: 'rgba(216, 207, 188, 0.4)', drawTicks: false };
    const yGrid = { display: opts.yGrid !== false, color: 'rgba(216, 207, 188, 0.4)', drawTicks: false };
    const tick = { color: COLOR.ink3, font: { family: FONT_SANS, size: 11.5 } };
    return {
      x: {
        grid: xGrid,
        border: { color: COLOR.line },
        ticks: tick,
        title: opts.xTitle ? { display: true, text: opts.xTitle, color: COLOR.ink3, font: { family: FONT_SANS, size: 12, weight: '500' }, padding: { top: 8 } } : undefined,
        min: opts.xMin,
        max: opts.xMax,
      },
      y: {
        grid: yGrid,
        border: { display: false },
        ticks: tick,
        title: opts.yTitle ? { display: true, text: opts.yTitle, color: COLOR.ink3, font: { family: FONT_SANS, size: 12, weight: '500' }, padding: { bottom: 8 } } : undefined,
        suggestedMin: opts.yMin,
        suggestedMax: opts.yMax,
      },
    };
  }

  // ===========================================================================
  // Per-seed SDR distribution — visualises the output-collapse regime
  // ===========================================================================
  const collapseCtx = document.getElementById('chart-collapse');
  if (collapseCtx) {
    // From docs/EXPERIMENT_LOG.md, Phase 2 and Phase 3
    // Each point = one training run (seed, model, SDR)
    const points = {
      'C-SE (Proposed)': [
        { seed: 42, sdr: 2.79, sir: 5.20 },
        { seed: 43, sdr: 2.74, sir: 5.73 },
        { seed: 44, sdr: 2.74, sir: 5.71 },
        { seed: 45, sdr: 1.62, sir: 19.55 },
        { seed: 46, sdr: 1.64, sir: 20.57 },
      ],
      'no-SE matched': [
        { seed: 42, sdr: 1.59, sir: 20.66 },
        { seed: 43, sdr: 2.66, sir: 5.92 },
        { seed: 44, sdr: 1.59, sir: 20.80 },
        { seed: 45, sdr: 1.62, sir: 22.19 },
        { seed: 46, sdr: 1.60, sir: 20.51 },
      ],
      'Real matched': [
        { seed: 42, sdr: 2.66, sir: 5.73 },
        { seed: 43, sdr: 2.65, sir: 5.35 },
        { seed: 44, sdr: 2.59, sir: 5.70 },
        { seed: 45, sdr: 1.56, sir: 20.23 },
        { seed: 46, sdr: 1.55, sir: 21.00 },
      ],
    };

    const datasets = Object.entries(points).map(([model, pts], idx) => {
      const color = idx === 0 ? COLOR.gold : (idx === 1 ? COLOR.blue : COLOR.purple);
      return {
        label: model,
        data: pts.map(p => ({ x: p.sdr, y: p.sir, _seed: p.seed })),
        backgroundColor: color + 'cc',
        borderColor: color,
        pointStyle: 'circle',
        pointRadius: 9,
        pointHoverRadius: 12,
        borderWidth: 1.5,
      };
    });

    new Chart(collapseCtx, {
      type: 'scatter',
      data: { datasets },
      options: {
        scales: makeScales({
          xTitle: 'SDR (dB, all modulation pairs averaged, SNR = 10 dB)',
          yTitle: 'SIR (dB)',
          xMin: 1.2, xMax: 3.2, yMin: 0, yMax: 25,
        }),
        plugins: {
          legend: { position: 'top', align: 'end' },
          tooltip: {
            callbacks: {
              label: (ctx) => {
                const p = ctx.raw;
                return `${ctx.dataset.label} · seed ${p._seed}: SDR=${p.x.toFixed(2)} dB, SIR=${p.y.toFixed(2)} dB`;
              },
            },
          },
        },
      },
      plugins: [{
        id: 'cluster-shading',
        beforeDatasetsDraw(chart) {
          const { ctx, scales, chartArea } = chart;
          if (!chartArea) return;
          ctx.save();

          // Working cluster (high SDR ~ 2.6-2.8, low SIR ~ 5-7)
          const wX0 = scales.x.getPixelForValue(2.45);
          const wX1 = scales.x.getPixelForValue(2.95);
          const wY0 = scales.y.getPixelForValue(7.5);
          const wY1 = scales.y.getPixelForValue(4.5);
          ctx.fillStyle = 'rgba(200, 150, 74, 0.10)';
          ctx.fillRect(wX0, wY1, wX1 - wX0, wY0 - wY1);
          ctx.strokeStyle = 'rgba(200, 150, 74, 0.5)';
          ctx.setLineDash([5, 4]);
          ctx.lineWidth = 1.2;
          ctx.strokeRect(wX0, wY1, wX1 - wX0, wY0 - wY1);
          ctx.fillStyle = COLOR.gold2;
          ctx.font = `600 12px ${FONT_SANS}`;
          ctx.textAlign = 'center';
          ctx.fillText('Working regime', (wX0 + wX1) / 2, wY0 + 16);

          // Collapse cluster (low SDR ~ 1.5-1.7, high SIR ~ 19-22)
          const cX0 = scales.x.getPixelForValue(1.4);
          const cX1 = scales.x.getPixelForValue(1.8);
          const cY0 = scales.y.getPixelForValue(23);
          const cY1 = scales.y.getPixelForValue(18);
          ctx.fillStyle = 'rgba(198, 107, 90, 0.10)';
          ctx.fillRect(cX0, cY1, cX1 - cX0, cY0 - cY1);
          ctx.strokeStyle = 'rgba(198, 107, 90, 0.5)';
          ctx.strokeRect(cX0, cY1, cX1 - cX0, cY0 - cY1);
          ctx.fillStyle = COLOR.terra;
          ctx.textAlign = 'center';
          ctx.fillText('Collapse regime', (cX0 + cX1) / 2, cY1 - 6);
          ctx.restore();
        },
      }],
    });
  }

  // ---------------------------------------------------------------------------
  // KaTeX auto-render
  // ---------------------------------------------------------------------------
  if (window.renderMathInElement) {
    window.renderMathInElement(document.body, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '$', right: '$', display: false },
      ],
      throwOnError: false,
    });
  }

  // ---------------------------------------------------------------------------
  // Sticky nav shadow on scroll
  // ---------------------------------------------------------------------------
  const topnav = document.getElementById('topnav');
  if (topnav) {
    const onScroll = () => {
      if (window.scrollY > 4) topnav.classList.add('is-stuck');
      else topnav.classList.remove('is-stuck');
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
  }

})();
