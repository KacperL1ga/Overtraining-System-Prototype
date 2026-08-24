import test from 'node:test';
import assert from 'node:assert/strict';
import { parseMedicalCsv, evaluateMetrics, generateAdvice } from './app.mjs';

test('parseMedicalCsv reads last CSV row with required fields', () => {
  const csv = [
    'resting_hr,hrv,sleep_hours,training_load,soreness,mood',
    '58,68,7.5,55,4,7',
    '64,50,5.5,88,8,4'
  ].join('\n');

  const data = parseMedicalCsv(csv);
  assert.equal(data.resting_hr, 64);
  assert.equal(data.hrv, 50);
  assert.equal(data.sleep_hours, 5.5);
  assert.equal(data.training_load, 88);
  assert.equal(data.soreness, 8);
  assert.equal(data.mood, 4);
});

test('evaluateMetrics returns bounded percentages', () => {
  const metrics = evaluateMetrics({
    resting_hr: 66,
    hrv: 48,
    sleep_hours: 5,
    training_load: 90,
    soreness: 8,
    mood: 3
  });

  assert.ok(metrics.overtrainingProbability >= 0 && metrics.overtrainingProbability <= 100);
  assert.ok(metrics.regenerationLevel >= 0 && metrics.regenerationLevel <= 100);
  assert.ok(metrics.overtrainingProbability > 70);
  assert.ok(metrics.regenerationLevel < 50);
});

test('generateAdvice returns recommendations for low and high values', () => {
  const advice = generateAdvice({ overtrainingProbability: 75, regenerationLevel: 35 });
  assert.equal(advice.length, 2);
  assert.match(advice[0], /Wysokie ryzyko przetrenowania/);
  assert.match(advice[1], /Niski poziom regeneracji/);
});
